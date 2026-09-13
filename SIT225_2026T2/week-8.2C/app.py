"""
app.py - SIT225 Task 5C entry point.

Streams accel_x/accel_y/accel_z from Arduino IoT Cloud into a smooth,
live-updating Plotly Dash chart using the generic smooth_stream wrapper.

"""

from arduino_cloud_client import ArduinoMQTTSource
from smooth_stream import smooth_stream_plot

CHANNELS = ["accel_x", "accel_y", "accel_z"]

source = ArduinoMQTTSource(channels=CHANNELS)
source.start()

plotter = smooth_stream_plot(
    data_source=source.read_latest,
    channels=CHANNELS,
    window_size=200,          # last 200 points shown (rolling window)
    update_interval_ms=100,   # redraw every 100ms -> smooth to the eye
    poll_interval_s=0.05,     # cheap: just reads the local cache, MQTT
                               # already pushed the real update in the background
    title="Live Smartphone Accelerometer Data",
)

plotter.start()
app = plotter.build_app()

if __name__ == "__main__":
    app.run(debug=True)
