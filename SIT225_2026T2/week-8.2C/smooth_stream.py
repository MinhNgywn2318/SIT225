"""
smooth_stream.py

SIT225 Task 5C - reusable wrapper for smooth, real-time Plotly Dash graph
updates from ANY continuous data source (not just accelerometer data).

Design summary (see Q1/Q2 write-up for full justification):

1. Ingestion is decoupled from rendering. A background thread continuously
   polls `data_source()` and timestamps every real sample it receives into
   a small rolling raw buffer (one per channel). Real sensor data almost
   never arrives at a perfectly steady rate - Arduino Cloud in particular
   only publishes a new value every so often - so ingestion must never be
   tied 1:1 to what gets drawn on screen.

2. Rendering runs on its own fixed-tick dcc.Interval, completely
   independent of how often real samples arrive. On every tick it computes
   a LINEARLY INTERPOLATED value between the two real samples that bracket
   "now minus a small render delay" for each channel. This is the key fix
   for the classic symptom of live dashboards: without interpolation, a
   slow-arriving sensor produces a staircase (flat segments that suddenly
   jump), because the same last known value gets redrawn on every tick
   until the next real sample shows up. Interpolating between the last two
   real points turns that staircase into a continuous line.

   A small render delay (default 300ms) is deliberately introduced so that
   the interpolation always has two REAL points to interpolate between
   (rather than guessing/extrapolating past the latest point, which can
   overshoot). This trades a small amount of latency for a visibly smooth
   curve - a design trade-off worth stating explicitly in the report.

3. Drawing uses Dash's Patch() object to update only the trace x/y arrays
   in-place, instead of rebuilding and re-sending a whole new Figure on
   every tick. This avoids the "flash/redraw" effect the task brief
   describes.

4. A rolling window (maxlen deque) keeps only the last N rendered points,
   so the graph pans smoothly rather than growing unbounded.

5. Everything is generic: the wrapper only needs a callable that returns a
   dict of the latest sample per channel, so it works for accelerometer
   data, temperature, or any other stream a peer plugs in.

Usage
-----
    from smooth_stream import smooth_stream_plot

    def my_sensor_read():
        # return None if there's no new sample yet, otherwise e.g.
        return {"x": 0.12, "y": -0.03, "z": 9.81}

    plotter = smooth_stream_plot(
        data_source=my_sensor_read,
        channels=["x", "y", "z"],
        window_size=200,          # rendered points kept on screen
        update_interval_ms=100,   # how often the graph redraws
        poll_interval_s=0.05,     # how often data_source() is checked
        render_delay_s=0.3,       # interpolation buffer window
        title="My Live Sensor Data",
    )
    plotter.start()
    app = plotter.build_app()
    app.run(debug=True)
"""

import threading
import time
from collections import deque

import dash
from dash import Input, Output, Patch, dcc, html


class SmoothStreamPlotter:
    """Wraps a continuous data source into a self-updating, smooth Dash chart."""

    def __init__(
        self,
        data_source,
        channels,
        window_size=200,
        update_interval_ms=100,
        poll_interval_s=0.05,
        render_delay_s=0.3,
        raw_buffer_size=200,
        title="Live Sensor Data",
    ):
        """
        Parameters
        ----------
        data_source : callable
            Zero-argument callable returning the latest sample as a dict,
            e.g. {'x': 0.12, 'y': -0.03, 'z': 9.81}, or None if no new
            sample is available yet.
        channels : list[str]
            Names of the series to plot; must match keys in data_source()'s
            returned dict.
        window_size : int
            Number of most recent RENDERED points kept on screen.
        update_interval_ms : int
            How often (ms) the Dash callback ticks to redraw.
        poll_interval_s : float
            How often (s) the background thread checks data_source() for a
            new real sample.
        render_delay_s : float
            How far behind "now" the rendered curve trails, so interpolation
            always has two real points to work with. Larger = smoother but
            more latency.
        raw_buffer_size : int
            How many real (unsmoothed) samples to keep per channel for
            interpolation purposes.
        title : str
            Chart title.
        """
        self.data_source = data_source
        self.channels = list(channels)
        self.window_size = window_size
        self.update_interval_ms = update_interval_ms
        self.poll_interval_s = poll_interval_s
        self.render_delay_s = render_delay_s
        self.title = title

        self._lock = threading.Lock()
        # Raw, timestamped real samples - the ground truth used for interpolation.
        self._raw = {ch: deque(maxlen=raw_buffer_size) for ch in self.channels}
        # Rendered (possibly interpolated) points actually shown on screen.
        self._render_t = deque(maxlen=window_size)
        self._render_y = {ch: deque(maxlen=window_size) for ch in self.channels}

        self._start_time = None
        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._app = None

    def _poll_loop(self):
        while not self._stop_event.is_set():
            sample = self.data_source()
            if sample is not None:
                now = time.time()
                with self._lock:
                    if self._start_time is None:
                        self._start_time = now
                    for ch in self.channels:
                        self._raw[ch].append((now, sample.get(ch)))
            time.sleep(self.poll_interval_s)

    @staticmethod
    def _interpolate(raw_points, t):
        """Linear interpolation of raw_points (list of (t, v)) at time t."""
        if not raw_points:
            return None
        if t <= raw_points[0][0]:
            return raw_points[0][1]
        if t >= raw_points[-1][0]:
            return raw_points[-1][1]
        prev_t, prev_v = raw_points[0]
        for cur_t, cur_v in raw_points:
            if cur_t >= t:
                if cur_t == prev_t:
                    return cur_v
                frac = (t - prev_t) / (cur_t - prev_t)
                return prev_v + (cur_v - prev_v) * frac
            prev_t, prev_v = cur_t, cur_v
        return raw_points[-1][1]

    def start(self):
        """Start the background polling thread. Call before app.run()."""
        self._poll_thread.start()

    def stop(self):
        """Stop the background polling thread."""
        self._stop_event.set()

    def build_app(self):
        """Builds and returns a ready-to-run Dash app with the smooth chart wired up."""
        initial_figure = {
            "data": [
                {"x": [], "y": [], "mode": "lines", "name": ch}
                for ch in self.channels
            ],
            "layout": {
                "title": self.title,
                "uirevision": "constant",  # keeps zoom/pan state across updates
                "xaxis": {"title": "Time (s)"},
                "yaxis": {"title": "Value"},
            },
        }

        app = dash.Dash(__name__)
        app.layout = html.Div(
            [
                html.H2(self.title),
                dcc.Graph(id="smooth-stream-graph", figure=initial_figure),
                dcc.Interval(
                    id="smooth-stream-interval",
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
            ]
        )

        @app.callback(
            Output("smooth-stream-graph", "figure"),
            Input("smooth-stream-interval", "n_intervals"),
        )
        def _update(_n_intervals):
            patched = Patch()
            now = time.time()
            with self._lock:
                if self._start_time is None:
                    return patched  # no data has arrived yet
                render_time = now - self.render_delay_s
                elapsed = render_time - self._start_time
                if elapsed < 0:
                    return patched

                self._render_t.append(elapsed)
                for ch in self.channels:
                    y = self._interpolate(list(self._raw[ch]), render_time)
                    self._render_y[ch].append(y)

                t = list(self._render_t)
                for i, ch in enumerate(self.channels):
                    patched["data"][i]["x"] = t
                    patched["data"][i]["y"] = list(self._render_y[ch])
            return patched

        self._app = app
        return app


def smooth_stream_plot(data_source, channels, **kwargs):
    """
    Convenience factory function - the "one call" API described in the task
    brief (Step 4): peers can get a smooth, real-time Dash chart for any
    continuous data source without knowing anything about buffering,
    threading, interpolation, or Patch() internally.

    Returns a SmoothStreamPlotter. Call .start() then .build_app().run(...).
    """
    return SmoothStreamPlotter(data_source, channels, **kwargs)
