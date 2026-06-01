"""
GRAPH-PLOT-PRESENTATION.py  —  Clean presentation figures
==========================================================
Produces two publication-clean figures from one or more gauge_log_*.csv files:

  1. Log-log plot: H-P predicted vs measured flow vs deltaP
  2. Ratio plot:   Q_measured / Q_HP vs deltaP

Legends show only "H-P predicted" and "Measured flow" — no filenames.
If multiple CSV files are loaded, each dataset gets its own colour but
still uses clean labels (Dataset 1, Dataset 2, …).

Usage
-----
  python GRAPH-PLOT-PRESENTATION.py                   # opens a file dialog
  python GRAPH-PLOT-PRESENTATION.py gauge_log_*.csv   # one or more files

Requirements
------------
  pip install matplotlib numpy scipy
"""

import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FONT SIZES
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":        32,
    "axes.titlesize":   40,
    "axes.labelsize":   36,
    "xtick.labelsize":  30,
    "ytick.labelsize":  30,
    "legend.fontsize":  28,
    "figure.titlesize": 44,
})

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CAPILLARY_RADIUS_M     = 25e-6       # m  (50 um ID)
CAPILLARY_LENGTH_M     = 2.0         # m
ETA_N2_PA_S            = 1.76e-5     # Pa.s  (N2 at ~20 C)
P_ATM_PA               = 101_325     # Pa

ELFLOW_N2_CORRECTION   = 1.0 / 1.40  # He-indicated -> actual N2
ELFLOW_SPIKE_THRESHOLD = 0.15        # mln/min He-indicated; above = spike
MIN_ELFLOW_SAMPLES     = 3           # min samples per bin to plot
BIN_WIDTH_MBAR         = 20.0        # mbar bin width for EL-Flow averaging

X_MIN_MBAR = 500.0
X_MAX_MBAR = 5000.0

# Colours: first entry = H-P line, remaining = measured datasets
HP_COLOUR      = "#4a90d9"   # clean blue for H-P predicted
DATA_COLOURS   = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#1f77b4"]

# ─────────────────────────────────────────────────────────────────────────────
# HAGEN-POISEUILLE
# ─────────────────────────────────────────────────────────────────────────────

def hp_flow_mln_min(P1_mbar):
    """Q = pi r^4 P1^2 / (16 eta L P_atm)  ->  mln/min actual N2."""
    P1_pa = P1_mbar * 100.0
    Q_m3s = (math.pi * CAPILLARY_RADIUS_M**4 * P1_pa**2
             / (16.0 * ETA_N2_PA_S * CAPILLARY_LENGTH_M * P_ATM_PA))
    return Q_m3s * 1e6 * 60

# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    """Returns (timestamps, pressures_mbar, elflow_he_mln_min)."""
    timestamps, pressures, elflows = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t  = datetime.fromisoformat(row["timestamp"])
                p  = float(row["upstream_pressure_bar"]) * 1000.0
                ef = row.get("elflow_mln_min", "").strip()
                ef = float(ef) if ef else None
                timestamps.append(t)
                pressures.append(p)
                elflows.append(ef)
            except (ValueError, KeyError):
                pass
    return timestamps, pressures, elflows


def load_all(paths):
    """Load and process all CSV files. Returns list of dataset dicts."""
    datasets = []
    for idx, path in enumerate(paths):
        timestamps, pressures, elflows = load_csv(path)
        if not pressures:
            print(f"  [!] No valid data in {path}")
            continue

        pressures = np.array(pressures)
        hp_flows  = np.array([hp_flow_mln_min(p) for p in pressures])
        sort_idx  = np.argsort(pressures)

        # EL-Flow: bin, average, correct He->N2
        bins = np.arange(pressures.min(), pressures.max() + BIN_WIDTH_MBAR, BIN_WIDTH_MBAR)
        ef_p, ef_q = [], []
        for b in bins:
            mask    = (pressures >= b) & (pressures < b + BIN_WIDTH_MBAR)
            ef_vals = [elflows[i] for i in range(len(elflows))
                       if mask[i] and elflows[i] is not None
                       and 0 < elflows[i] < ELFLOW_SPIKE_THRESHOLD]
            if len(ef_vals) >= MIN_ELFLOW_SAMPLES:
                ef_p.append(b + BIN_WIDTH_MBAR / 2)
                ef_q.append(np.mean(ef_vals) * ELFLOW_N2_CORRECTION)

        # Clean label: single dataset gets no suffix; multiples get numbered
        label = "Q_measured / Q_HP" if len(paths) == 1 else f"Q_measured / Q_HP  ({idx + 1})"

        ds = {
            "label":     label,
            "col":       DATA_COLOURS[idx % len(DATA_COLOURS)],
            "p_sorted":  pressures[sort_idx],
            "hp_sorted": hp_flows[sort_idx],
            "ef_p":      np.array(ef_p),
            "ef_q":      np.array(ef_q),
            "stats":     None,
        }

        if len(ef_p) >= 3:
            ds["stats"] = compute_statistics(ds)

        datasets.append(ds)
        print(f"  {Path(path).stem}: {len(pressures)} rows, "
              f"P {pressures.min():.1f}–{pressures.max():.1f} mbar, "
              f"{len(ef_p)} EL-Flow bins")
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(ds):
    ef_p = ds["ef_p"]
    ef_q = ds["ef_q"]
    hp_q = np.array([hp_flow_mln_min(p) for p in ef_p])
    n    = len(ef_q)

    slope  = np.dot(hp_q, ef_q) / np.dot(hp_q, hp_q)
    ss_res = np.sum((ef_q - slope * hp_q) ** 2)
    ss_tot = np.sum(ef_q ** 2)
    r2     = 1.0 - ss_res / ss_tot

    se      = np.sqrt(ss_res / (n - 1)) / np.sqrt(np.dot(hp_q, hp_q))
    t_slope = slope / se
    p_reg   = 2 * stats.t.sf(abs(t_slope), df=n - 1)

    ratios     = ef_q / hp_q
    ratio_mean = ratios.mean()
    ratio_std  = ratios.std(ddof=1)

    return {
        "n":          n,
        "slope":      slope,
        "r2":         r2,
        "p_reg":      p_reg,
        "ratio_mean": ratio_mean,
        "ratio_std":  ratio_std,
        "ratios":     ratios,
        "hp_q":       hp_q,
        "ef_q":       ef_q,
        "ef_p":       ef_p,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: LOG-LOG PLOT
# ─────────────────────────────────────────────────────────────────────────────

def make_loglog_figure(datasets):
    fig, ax = plt.subplots(figsize=(26, 16))

    # H-P reference line — single line, clean label
    p_ref = np.geomspace(X_MIN_MBAR, X_MAX_MBAR, 500)
    q_ref = np.array([hp_flow_mln_min(p) for p in p_ref])
    ax.plot(p_ref, q_ref,
            color=HP_COLOUR, linewidth=2.5, zorder=3,
            label="H-P predicted")

    # Measured data per dataset
    for ds in datasets:
        if len(ds["ef_p"]):
            ax.scatter(ds["ef_p"], ds["ef_q"],
                       color=ds["col"], s=60, zorder=5, marker="o",
                       label=ds["label"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X_MIN_MBAR, X_MAX_MBAR)
    ax.set_xticks([500, 1000, 2000, 5000])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.set_xlabel("\u0394P  (mbar)", fontsize=36, labelpad=8)
    ax.set_ylabel("Flow rate  (mln/min, actual N\u2082)", fontsize=36, labelpad=8)
    ax.set_title("Capillary conductance: Hagen-Poiseuille vs measured",
                 fontsize=40, pad=14)

    ax.grid(which="major", color="#dddddd", linewidth=0.8, zorder=0)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.4, zorder=0)
    ax.legend(fontsize=28, loc="upper left")

    note = (f"50 \u03bcm ID \u00d7 2.0 m  |  N\u2082  \u03b7 = {ETA_N2_PA_S:.2e} Pa\u00b7s  |  "
            f"EL-Flow \u00d7{ELFLOW_N2_CORRECTION:.3f} (He\u2192N\u2082)  |  "
            f"\u0394P = P\u2081  (P\u2082 \u2248 0)")
    fig.text(0.5, 0.01, note, ha="center", color="#888888", fontsize=24)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: RATIO PLOT
# ─────────────────────────────────────────────────────────────────────────────

def make_ratio_figure(datasets):
    fig, ax = plt.subplots(figsize=(26, 14))

    any_data = False

    for ds in datasets:
        st = ds.get("stats")
        if st is None:
            continue

        ratios = st["ratios"]
        ef_p   = st["ef_p"]

        ax.scatter(ef_p, ratios,
                   color=ds["col"], s=60, zorder=5, marker="o",
                   label=ds["label"])
        ax.plot(ef_p, ratios,
                color=ds["col"], linewidth=1.2, linestyle="--",
                alpha=0.6, zorder=4)

        # Mean line
        mean_r = st["ratio_mean"]
        ax.axhline(mean_r, color=ds["col"], linewidth=1.5, linestyle=":",
                   alpha=0.8, zorder=3)

        any_data = True

    if not any_data:
        print("  [!] No paired EL-Flow / H-P data for ratio plot.")
        plt.close(fig)
        return

    # Ratio = 1.0 reference — drawn but not in legend
    ax.axhline(1.0, color="#aaaaaa", linewidth=1.4, linestyle="--", zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(X_MIN_MBAR, X_MAX_MBAR)
    ax.set_xticks([500, 1000, 2000, 5000])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Crop y-axis to data range with padding; annotate where ratio=1 lies
    all_ratios = np.concatenate([ds["stats"]["ratios"]
                                 for ds in datasets if ds.get("stats")])
    y_lo = max(0.0, all_ratios.min() - 0.05)
    y_hi = all_ratios.max() + 0.05
    ax.set_ylim(y_lo, y_hi)

    ax.annotate(
        "ratio = 1.0  (perfect agreement)  \u2191",
        xy=(X_MAX_MBAR * 0.75, y_hi),
        xytext=(X_MAX_MBAR * 0.75, y_hi - (y_hi - y_lo) * 0.08),
        fontsize=26, color="#888888", ha="center", va="top",
        annotation_clip=False,
    )

    ax.set_xlabel("\u0394P  (mbar)", fontsize=36, labelpad=8)
    ax.set_ylabel("Flow ratio", fontsize=36, labelpad=8)
    ax.set_title("Ratio plot: measured / H-P predicted flow vs pressure",
                 fontsize=40, pad=14)

    ax.grid(which="major", color="#dddddd", linewidth=0.8, zorder=0)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.4, zorder=0)
    ax.legend(fontsize=28, loc="upper left")

    note = (f"50 \u03bcm ID \u00d7 2.0 m  |  N\u2082  \u03b7 = {ETA_N2_PA_S:.2e} Pa\u00b7s  |  "
            f"EL-Flow \u00d7{ELFLOW_N2_CORRECTION:.3f} (He\u2192N\u2082)  |  "
            f"dotted = mean ratio")
    fig.text(0.5, 0.01, note, ha="center", color="#888888", fontsize=24)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(paths):
    print(f"Loading {len(paths)} file(s)...")
    datasets = load_all(paths)
    if not datasets:
        print("No valid data found.")
        return

    make_loglog_figure(datasets)
    make_ratio_figure(datasets)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            paths = filedialog.askopenfilenames(
                title="Select gauge_log CSV file(s)",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not paths:
                print("No files selected. Exiting.")
                sys.exit(0)
        except ImportError:
            print("Usage: python GRAPH-PLOT-PRESENTATION.py <gauge_log.csv> [...]")
            sys.exit(1)

    main(paths)
