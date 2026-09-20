"""
Build annotations.csv (filename,label) and a gallery.html to review the images quickly.

Because you perform the activities in blocks, you can pre-fill labels by sequence number
in BLOCKS below. Pre-filled labels are only a starting point: open gallery.html and
check EVERY image, then fix wrong rows in annotations.csv (this is the Q3 work).

Label meaning: 0 = no-activity, 1 = activity 1, 2 = activity 2 (rename in analyze.py).
"""
import csv
import os
import re

DATA_DIR = "data"
# (first_seq, last_seq, label) - edit to match your session notes. Leave empty to label by hand.
BLOCKS = [
    (1, 21, 0),
    (22, 41, 1),
    (42, 61, 2),
    (81, 100, 0),
    (105, 120, 1),
    (122, 169, 2),
]


def block_label(seq):
    for lo, hi, lab in BLOCKS:
        if lo <= seq <= hi:
            return lab
    return ""


files = sorted((f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")),
               key=lambda f: int(re.match(r"(\d+)_", f).group(1)))

with open("annotations.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label"])
    for name in files:
        seq = int(re.match(r"(\d+)_", name).group(1))
        w.writerow([name.replace(".jpg", ""), block_label(seq)])

with open("gallery.html", "w") as f:
    f.write("<html><body style='font-family:sans-serif'><div style='display:flex;flex-wrap:wrap'>")
    for name in files:
        f.write(f"<div style='margin:4px;width:220px'><img src='{DATA_DIR}/{name}' width='220'>"
                f"<br><small>{name}</small></div>")
    f.write("</div></body></html>")

print(f"{len(files)} images -> annotations.csv, gallery.html")