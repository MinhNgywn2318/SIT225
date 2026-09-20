"""
SIT225 8.3D - Capture phone accelerometer data + webcam image every WINDOW_SEC seconds.

Flow: phone -> Arduino IoT Cloud -> this script.
Every WINDOW_SEC seconds of data:
  - save  data/<seq>_<yyyymmddHHMMSS>.csv  (t_sec, x, y, z)
  - save  data/<seq>_<yyyymmddHHMMSS>.jpg  (webcam frame)
  - update the Plotly Dash page (graph + image)

Install:  pip install arduino-iot-cloud dash plotly opencv-python
Fill in DEVICE_ID, SECRET_KEY and the variable names you used in Week 8.
"""
import base64
import os
import re
import threading
import time
from datetime import datetime

import cv2
import plotly.graph_objects as go
from arduino_iot_cloud import ArduinoCloudClient
from dash import Dash, Input, Output, dcc, html

# ---------- settings (edit these) ----------
DEVICE_ID = "3bd7beb5-dd1a-4b09-8628-9ddbe19e84f8"
SECRET_KEY = "Ie95aB4TiU!e54@Ff3SfxBby3"
VAR_X, VAR_Y, VAR_Z = "accel_x", "accel_y", "accel_z"  # cloud variable names from Week 8
WINDOW_SEC = 10          # length of one window; justify your choice in Q1
CAMERA_INDEX = 0
OUT_DIR = "data"
# -------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

lock = threading.Lock()
latest = {"x": 0.0, "y": 0.0, "z": 0.0}
have = {"x": False, "y": False, "z": False}
buf = []                 # list of (t_sec, x, y, z) for the current window
win_start = None         # time.time() when the current window began
state = {"name": "waiting for data...", "t": [], "x": [], "y": [], "z": [], "img": None}


def next_seq():
    """Continue numbering after any files already in OUT_DIR (safe on restart)."""
    nums = [int(m.group(1)) for f in os.listdir(OUT_DIR)
            if (m := re.match(r"(\d+)_\d{14}\.csv$", f))]
    return max(nums, default=0) + 1


seq = next_seq()
cam = cv2.VideoCapture(CAMERA_INDEX)
for _ in range(5):       # let the webcam auto-exposure settle
    cam.read()


def flush_window():
    """Close the current window: save CSV + JPG and publish to the dashboard."""
    global seq, buf, win_start
    ok, frame = cam.read()
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = f"{seq}_{stamp}"
    with open(os.path.join(OUT_DIR, name + ".csv"), "w") as f:
        f.write("t_sec,x,y,z\n")
        for t, x, y, z in buf:
            f.write(f"{t:.4f},{x},{y},{z}\n")
    img_b64 = None
    if ok:
        cv2.imwrite(os.path.join(OUT_DIR, name + ".jpg"), frame)
        _, enc = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(enc).decode()
    state.update(name=name, t=[r[0] for r in buf], x=[r[1] for r in buf],
                 y=[r[2] for r in buf], z=[r[3] for r in buf], img=img_b64)
    print(f"saved {name}  ({len(buf)} samples)")
    seq += 1
    buf = []
    win_start = None


def add_sample():
    """Called on every incoming value: append (t, x, y, z) to the window."""
    global win_start
    with lock:
        if not all(have.values()):
            return
        now = time.time()
        if win_start is None:
            win_start = now
        buf.append((now - win_start, latest["x"], latest["y"], latest["z"]))
        if now - win_start >= WINDOW_SEC:
            flush_window()


def make_cb(axis):
    def cb(client, value):
        if value is None:
            return
        latest[axis] = value
        have[axis] = True
        print(f"cloud update: {axis} = {value}")   # debug - remove once it's working
        add_sample()          # sample on ANY axis update, not just x
    return cb


def run_cloud():
    client = ArduinoCloudClient(device_id=DEVICE_ID, username=DEVICE_ID, password=SECRET_KEY)
    client.register(VAR_X, value=None, on_write=make_cb("x"))
    client.register(VAR_Y, value=None, on_write=make_cb("y"))
    client.register(VAR_Z, value=None, on_write=make_cb("z"))
    client.start()


app = Dash(__name__)
app.layout = html.Div([
    html.H3("Accelerometer + activity image"),
    html.Div(id="title"),
    dcc.Graph(id="graph"),
    html.Img(id="img", style={"maxWidth": "480px"}),
    dcc.Interval(id="tick", interval=1000),
])


@app.callback(Output("graph", "figure"), Output("img", "src"), Output("title", "children"),
              Input("tick", "n_intervals"))
def refresh(_):
    fig = go.Figure()
    for axis in "xyz":
        fig.add_trace(go.Scatter(x=state["t"], y=state[axis], mode="lines", name=axis))
    fig.update_layout(xaxis_title="seconds in window", yaxis_title="acceleration")
    src = f"data:image/jpeg;base64,{state['img']}" if state["img"] else ""
    return fig, src, f"Latest window: {state['name']}"


if __name__ == "__main__":
    threading.Thread(target=run_cloud, daemon=True).start()
    app.run(debug=False)
