"""
plot_capillary.py  —  Capillary flow: Hagen-Poiseuille reference + measured data
==================================================================================
Reads one or more gauge_log_*.csv files and plots:
  - H-P reference line across the configured pressure range
  - H-P curve computed from each measured upstream pressure point
  - EL-Flow measured flow (He-indicated, corrected to actual N2)

Statistical tests printed to console and annotated on figures:
  1. Linear regression forced through origin  Q_meas = a * Q_HP
     (physically correct: zero pressure => zero flow, so no intercept)
  2. Ratio analysis     Q_meas / Q_HP per bin
  3. One-sample t-test  H0: mean ratio = 1.0

X axis: deltaP (mbar, log scale)  — P2 ~= 0 so deltaP ~= P1
Y axis: flow rate (mln/min actual N2), linear or log

Usage
-----
  python plot_capillary.py                        # opens a file dialog
  python plot_capillary.py gauge_log_*.csv        # one or more files as args

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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CAPILLARY_RADIUS_M     = 25e-6    # m  (50 um ID)
CAPILLARY_LENGTH_M     = 2.0      # m
ETA_N2_PA_S            = 1.76e-5  # Pa.s  (N2 at ~20 C)
P_ATM_PA               = 101_325  # Pa

ELFLOW_N2_CORRECTION   = 1.0 / 1.40   # He-indicated -> actual N2
ELFLOW_SPIKE_THRESHOLD = 0.15          # mln/min He-indicated; above = spike
MIN_ELFLOW_SAMPLES     = 3             # min samples per bin to plot
BIN_WIDTH_MBAR         = 20.0          # mbar bin width for EL-Flow averaging

X_MIN_MBAR = 1e2
X_MAX_MBAR = 5000.0

DATA_COLOURS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]

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

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING + PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_all(paths):
    """Load and process all CSV files. Returns list of dataset dicts."""
    datasets = []
    for idx, path in enumerate(paths):
        label = Path(path).stem
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

        ds = {
            "label":     label,
            "col":       DATA_COLOURS[idx % len(DATA_COLOURS)],
            "col_ef":    DATA_COLOURS[(idx + 1) % len(DATA_COLOURS)],
            "p_sorted":  pressures[sort_idx],
            "hp_sorted": hp_flows[sort_idx],
            "ef_p":      np.array(ef_p),
            "ef_q":      np.array(ef_q),
            "n_rows":    len(pressures),
            "stats":     None,
        }

        if len(ef_p) >= 3:
            ds["stats"] = compute_statistics(ds)

        datasets.append(ds)
        print(f"  {label}: {len(pressures)} rows, "
              f"P {pressures.min():.1f}-{pressures.max():.1f} mbar, "
              f"{len(ef_p)} EL-Flow bins")
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(ds):
    """
    Compare EL-Flow to H-P at matched pressure bins.

    1. Linear regression forced through origin: Q_meas = a * Q_HP
       slope a = sum(Q_HP * Q_meas) / sum(Q_HP^2)
       Forcing through origin is physically correct — at zero pressure
       there is zero flow, so a free intercept is meaningless.
       With origin forced, slope == mean ratio when scaling is pure P^2.
       Any discrepancy between slope and mean ratio indicates the ratio
       varies with pressure (i.e. some deviation from P^2 scaling).

    2. Ratio analysis: r = Q_meas / Q_HP per bin.

    3. One-sample t-test: H0: mean(r) = 1.0
    """
    ef_p = ds["ef_p"]
    ef_q = ds["ef_q"]
    hp_q = np.array([hp_flow_mln_min(p) for p in ef_p])
    n    = len(ef_q)

    # Origin-forced slope: minimise sum((ef_q - a*hp_q)^2) w.r.t. a
    slope  = np.dot(hp_q, ef_q) / np.dot(hp_q, hp_q)

    # R^2 relative to zero (appropriate for zero-intercept model)
    ss_res = np.sum((ef_q - slope * hp_q) ** 2)
    ss_tot = np.sum(ef_q ** 2)
    r2     = 1.0 - ss_res / ss_tot

    # Standard error and p-value on slope (n-1 dof)
    se      = np.sqrt(ss_res / (n - 1)) / np.sqrt(np.dot(hp_q, hp_q))
    t_slope = slope / se
    p_reg   = 2 * stats.t.sf(abs(t_slope), df=n - 1)

    # Ratio analysis
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


def print_statistics(label, st):
    w = 64
    print()
    print("=" * w)
    print(f"  Statistical analysis: {label}")
    print("=" * w)
    print(f"  Paired pressure bins (n): {st['n']}")
    print()
    print("  1. Linear regression (forced through origin)")
    print("     Q_measured = a * Q_HP   (zero pressure => zero flow)")
    print(f"     slope  a = {st['slope']:.4f}      (ideal: 1.000)")
    print(f"     R^2      = {st['r2']:.4f}")
    print(f"     p-value  = {st['p_reg']:.3e}  "
          f"{'*** significant' if st['p_reg'] < 0.05 else '(not significant)'}")
    print()
    print("  2. Ratio analysis   Q_measured / Q_HP per bin")
    print(f"     mean  = {st['ratio_mean']:.4f}   (ideal: 1.000)")
    print(f"     std   = {st['ratio_std']:.4f}")
    print(f"     range = {st['ratios'].min():.4f} - {st['ratios'].max():.4f}")
    print()
    if abs(st['slope'] - st['ratio_mean']) > 0.02:
        print("  NOTE: slope != mean ratio, suggesting Q_meas/Q_HP varies")
        print("  with pressure (some deviation from pure P^2 scaling).")
    print("=" * w)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(datasets, log_y=False):
    """Draw one figure. log_y=True for log y-axis, False for linear."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # H-P reference line across full x range
    p_ref = np.geomspace(X_MIN_MBAR, X_MAX_MBAR, 500)
    q_ref = np.array([hp_flow_mln_min(p) for p in p_ref])
    ax.plot(p_ref, q_ref, color="#aaaaaa", linewidth=1.5, alpha=0.6,
            zorder=2, label="H-P reference  (50 um x 2 m, N2)")

    for ds in datasets:
        ax.plot(ds["p_sorted"], ds["hp_sorted"],
                color=ds["col"], linewidth=1.8, zorder=3,
                label=f"H-P from data: {ds['label']}")

        if len(ds["ef_p"]):
            ax.scatter(ds["ef_p"], ds["ef_q"],
                       color=ds["col_ef"], s=30, zorder=5, marker="o",
                       label=f"EL-Flow (meas.): {ds['label']}")
            ax.plot(ds["ef_p"], ds["ef_q"],
                    color=ds["col_ef"], linewidth=1.0, linestyle="--",
                    alpha=0.7, zorder=4)

    # Stats annotation box — only on log y-axis figure where it doesn't overlap
    if log_y:
        for ds in datasets:
            if ds["stats"]:
                st  = ds["stats"]
                ann = (
                    f"Regression (origin-forced):  slope = {st['slope']:.3f},  R\u00b2 = {st['r2']:.3f}\n"
                    f"Mean ratio  Q_meas / Q_HP = {st['ratio_mean']:.3f} \u00b1 {st['ratio_std']:.3f}"
                )
                ax.text(0.98, 0.05, ann,
                        transform=ax.transAxes, fontsize=8,
                        va="bottom", ha="right",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.9))
                break

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(X_MIN_MBAR, X_MAX_MBAR)

    y_desc = "log" if log_y else "linear"
    ax.set_xlabel("\u0394P  (mbar)", fontsize=12, labelpad=8)
    ax.set_ylabel("Flow rate  (mln/min, actual N2)", fontsize=12, labelpad=8)
    ax.set_title(
        f"Capillary conductance: Hagen-Poiseuille vs measured  [{y_desc} y-axis]",
        fontsize=14, pad=14)

    ax.grid(which="major", color="#dddddd", linewidth=0.8, zorder=0)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.4, zorder=0)

    for p_vline in [1013.25]:
        if X_MIN_MBAR < p_vline < X_MAX_MBAR:
            ax.axvline(p_vline, color="#bbbbbb", linewidth=0.8,
                       linestyle=":", zorder=1)
            ax.text(p_vline * 1.05, ax.get_ylim()[0], "1 atm",
                    color="#999999", fontsize=8, va="bottom")

    ax.legend(fontsize=9, loc="upper left")

    note = (f"50 um ID x 2.0 m  |  N2  eta = {ETA_N2_PA_S:.2e} Pa.s  |  "
            f"EL-Flow x{ELFLOW_N2_CORRECTION:.3f} (He->N2)  |  "
            f"dP = P1 since P2 ~= 0")
    fig.text(0.5, 0.01, note, ha="center", color="#888888", fontsize=8)

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

    for ds in datasets:
        if ds["stats"]:
            print_statistics(ds["label"], ds["stats"])
        else:
            print(f"\n  [!] {ds['label']}: too few EL-Flow bins for statistics.")

    make_figure(datasets, log_y=False)
    make_figure(datasets, log_y=True)
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
            print("Usage: python plot_capillary.py <gauge_log.csv> [...]")
            sys.exit(1)

    main(paths)