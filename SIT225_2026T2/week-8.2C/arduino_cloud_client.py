"""
arduino_cloud_client.py (MQTT version)

Connects to Arduino IoT Cloud over MQTT using DEVICE CREDENTIALS
(Device ID + Secret Key from a manually-configured device - e.g. the
"Steffi" device shown in your Arduino Cloud dashboard), NOT the REST API
Client ID/Secret from the API Keys page.

Why MQTT instead of REST polling:
    - Arduino Cloud PUSHES new values to us the moment they change
      (on_write callback), instead of us having to repeatedly ask "is
      there anything new yet?" over REST. This is lower latency and a
      better fit for the "smooth real-time update" goal of Task 5C - and
      it's worth citing as a design decision in your Q1 write-up.
    - It uses exactly the credentials Arduino Cloud already gave you for
      this device, no separate API Key needed.

This module exposes read_latest(), which has the exact same "return a
dict of the latest sample, or None" contract that smooth_stream.py's
data_source parameter expects - so smooth_stream.py and app.py did not
need to change at all when switching from REST to MQTT.
"""

import os
import threading

from arduino_iot_cloud import ArduinoCloudClient
from dotenv import load_dotenv

load_dotenv()


class ArduinoMQTTSource:
    def __init__(self, device_id=None, secret_key=None,
                 channels=("accel_x", "accel_y", "accel_z")):
        self.device_id = device_id or os.environ["ARDUINO_DEVICE_ID"]
        self.secret_key = secret_key or os.environ["ARDUINO_SECRET_KEY"]
        self.channels = list(channels)

        self._lock = threading.Lock()
        self._latest = {ch: None for ch in self.channels}
        # Tracks whether ANY channel has changed since the last read_latest()
        # call. Without this, read_latest() would keep returning the same
        # cached values on every poll even when nothing new has arrived,
        # which defeats smooth_stream.py's interpolation: it needs raw
        # samples timestamped at the moments real data actually changed,
        # not at every 50ms poll tick.
        self._dirty = False

        # username == device_id, password == secret_key: this is the
        # "basic auth" mode for manually-configured devices.
        self._client = ArduinoCloudClient(
            device_id=self.device_id,
            username=self.device_id,
            password=self.secret_key,
        )

        for ch in self.channels:
            self._client.register(ch, value=None, on_write=self._make_callback(ch))

        # client.start() runs its own asyncio loop and blocks forever,
        # so it needs its own background thread (Dash's server runs on
        # the main thread).
        self._thread = threading.Thread(target=self._client.start, daemon=True)

    def _make_callback(self, channel):
        def _on_write(client, value):
            with self._lock:
                self._latest[channel] = value
                self._dirty = True
        return _on_write

    def start(self):
        """Connect to Arduino Cloud and start listening for updates."""
        self._thread.start()

    def read_latest(self):
        """
        Matches the data_source() contract expected by
        smooth_stream.SmoothStreamPlotter: returns the latest known value
        of every channel, or None if either (a) not every channel has
        reported at least once yet, or (b) nothing has changed since the
        last call. Returning None on "no new data" (rather than repeating
        the last value) is what lets smooth_stream.py's interpolation see
        real, correctly-spaced timestamps.
        """
        with self._lock:
            if any(v is None for v in self._latest.values()):
                return None
            if not self._dirty:
                return None
            self._dirty = False
            return dict(self._latest)
