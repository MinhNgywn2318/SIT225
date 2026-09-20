"""
Pattern analysis for Q2 / Q3.

Reads annotations.csv + data/*.csv and produces in figures/:
  - features.csv            per-window features
  - feature_summary.csv     mean/std of each feature per class
  - examples_<class>.png    several raw windows per class
  - boxplots.png            feature distributions per class
  - spectra.png             mean FFT spectrum per class
  - suspect_labels.csv      windows whose data looks closer to another class (for Q3)

Install: pip install numpy pandas matplotlib
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data"
ANN = "annotations.csv"
OUT = "figures"
FS = 20            # resample rate (Hz) for FFT; pick close to your real sample rate
N_EXAMPLES = 4
NAMES = {0: "no-activity", 1: "activity 1", 2: "activity 2"}   # rename to real activities

os.makedirs(OUT, exist_ok=True)
ann_raw = pd.read_csv(ANN)
ann = ann_raw.dropna()
if ann.empty:
    raise SystemExit(
        f"annotations.csv has {len(ann_raw)} row(s) but none have a label filled in.\n"
        f"Open annotations.csv and make sure the 'label' column has 0/1/2 for the rows you want to analyze,\n"
        f"or edit BLOCKS in label_helper.py to cover your file's sequence numbers and rerun it."
    )
ann["label"] = ann["label"].astype(int)


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name + ".csv"))


def resample(df):
    t = np.arange(0, df["t_sec"].iloc[-1], 1 / FS)
    return t, {a: np.interp(t, df["t_sec"], df[a]) for a in "xyz"}


def features(df):
    t, ax = resample(df)
    mag = np.sqrt(ax["x"] ** 2 + ax["y"] ** 2 + ax["z"] ** 2)
    m = mag - mag.mean()
    spec = np.abs(np.fft.rfft(m)) ** 2
    freqs = np.fft.rfftfreq(len(m), 1 / FS)
    k = spec[1:].argmax() + 1 if len(spec) > 1 else 0
    return {
        "std_x": ax["x"].std(), "std_y": ax["y"].std(), "std_z": ax["z"].std(),
        "mag_std": m.std(), "mag_range": mag.max() - mag.min(),
        "mean_abs_diff": np.abs(np.diff(mag)).mean(),
        "dom_freq": freqs[k],
        "zero_cross": int((np.diff(np.sign(m)) != 0).sum()),
    }, freqs, spec


rows, spectra, missing = [], {}, []
for _, r in ann.iterrows():
    try:
        df = load(r["filename"])
    except FileNotFoundError:
        missing.append(r["filename"])
        continue
    f, freqs, spec = features(df)
    f.update(filename=r["filename"], label=r["label"])
    rows.append(f)
    spectra.setdefault(r["label"], []).append(np.interp(np.linspace(0, FS / 2, 60), freqs, spec))

if not rows:
    raise SystemExit(
        f"None of the {len(ann)} labeled filenames in annotations.csv matched a .csv file in '{DATA_DIR}/'.\n"
        f"First missing example: {missing[0] if missing else '(none)'}.csv\n"
        f"Check you are running this from the same folder as 'data/', and that filenames in\n"
        f"annotations.csv (e.g. '1_20260920184126') match the .csv files inside data/ exactly."
    )
if missing:
    print(f"warning: {len(missing)} labeled filename(s) had no matching .csv in {DATA_DIR}/, skipped: {missing[:5]}")

feat = pd.DataFrame(rows)
feat.to_csv(f"{OUT}/features.csv", index=False)
cols = [c for c in feat.columns if c not in ("filename", "label")]
feat.groupby("label")[cols].agg(["mean", "std"]).to_csv(f"{OUT}/feature_summary.csv")

# raw examples per class
for lab, name in NAMES.items():
    sub = feat[feat.label == lab].head(N_EXAMPLES)
    if sub.empty:
        continue
    fig, axes = plt.subplots(len(sub), 1, figsize=(9, 2.2 * len(sub)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax_, fn in zip(axes, sub.filename):
        d = load(fn)
        for a in "xyz":
            ax_.plot(d["t_sec"], d[a], label=a, lw=1)
        ax_.set_title(fn, fontsize=8)
    axes[0].legend(ncol=3, fontsize=7)
    fig.suptitle(f"Examples: {name}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/examples_{name.replace(' ', '_')}.png", dpi=130)
    plt.close(fig)

# boxplots
show = ["mag_std", "mag_range", "mean_abs_diff", "dom_freq"]
fig, axes = plt.subplots(1, len(show), figsize=(14, 3.5))
for ax_, c in zip(axes, show):
    try:
        ax_.boxplot([feat[feat.label == l][c] for l in NAMES], tick_labels=[NAMES[l] for l in NAMES])
    except TypeError:  # matplotlib < 3.9 uses 'labels' instead of 'tick_labels'
        ax_.boxplot([feat[feat.label == l][c] for l in NAMES], labels=[NAMES[l] for l in NAMES])
    ax_.set_title(c)
    ax_.tick_params(axis="x", labelrotation=20, labelsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/boxplots.png", dpi=130)
plt.close(fig)

# mean spectrum per class
plt.figure(figsize=(7, 3.5))
for lab, name in NAMES.items():
    if lab in spectra:
        plt.plot(np.linspace(0, FS / 2, 60), np.mean(spectra[lab], axis=0), label=name)
plt.xlabel("Hz"); plt.ylabel("power"); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/spectra.png", dpi=130)
plt.close()

# nearest-centroid check on standardised features: which windows look like another class?
use = ["mag_std", "mean_abs_diff", "dom_freq"]
z = (feat[use] - feat[use].mean()) / feat[use].std().replace(0, 1)
cent = z.groupby(feat.label).mean()
dist = pd.DataFrame({l: ((z - cent.loc[l]) ** 2).sum(axis=1) ** 0.5 for l in cent.index})
feat["nearest"] = dist.idxmin(axis=1)
feat["margin"] = dist.apply(lambda r: np.sort(r.values)[1] - np.sort(r.values)[0], axis=1)
sus = feat[(feat.nearest != feat.label) | (feat.margin < 0.5)]
sus[["filename", "label", "nearest", "margin"]].to_csv(f"{OUT}/suspect_labels.csv", index=False)
print(f"{len(feat)} windows analysed, {len(sus)} flagged in suspect_labels.csv")
