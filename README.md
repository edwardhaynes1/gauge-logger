# gauge-logger

Data acquisition system for capillary flow characterisation. Logs upstream pressure, vacuum chamber pressure, and mass flow rate to CSV for Hagen–Poiseuille validation experiments.

## Hardware

| Component | Description |
|---|---|
| LabJack U3 | DAQ device — analog inputs for pressure sensors |
| Bronkhorst F-200CV EL-Flow | Mass flow meter (He-calibrated, 3.0 mln/min FS) |
| 4–20 mA pressure transmitter | Upstream pressure, 0–10 bar, via 100 Ω shunt on FIO0 |
| Vacuum gauge | Log-linear analog output on FIO2 |

## Wiring (LabJack U3)

```
Transmitter +  →  VS  (5 V, pin 1)
Transmitter −  →  FIO0 + one leg of 100 Ω shunt
Shunt other leg →  GND (pin 2)

Vacuum gauge analog out  →  FIO2
```

## Software dependencies

```
pip install LabJackPython bronkhorst-propar==0.2.3 pyserial
```

For the analysis plotter:

```
pip install matplotlib numpy scipy
```

## Configuration

Edit the `CONFIGURATION` block at the top of `GAUGE_LOGGER.py`:

| Parameter | Default | Description |
|---|---|---|
| `ELFLOW_PORT` | `None` | COM port for EL-Flow. `None` = auto-detect |
| `ELFLOW_NODE` | `3` | Propar node address |
| `ELFLOW_FULLSCALE` | `3.0` | Full-scale flow in mln/min He |
| `LOG_INTERVAL_S` | `20` | Seconds between logged rows |
| `LABJACK_SAMPLE_HZ` | `10` | LabJack internal sampling rate (Hz) |
| `VACUUM_SLOPE` | `5.389` | Log-linear gauge calibration slope |
| `VACUUM_INTERCEPT` | `-11.329` | Log-linear gauge calibration intercept |

## Usage

```
python GAUGE_LOGGER.py
```

The EL-Flow is auto-detected on startup by scanning all available COM ports. The LabJack U3 is detected automatically by device type.

Output is written to a timestamped CSV in the same folder as the script:

```
gauge_log_YYYYMMDD_HHMMSS.csv
```

### CSV columns

| Column | Units | Description |
|---|---|---|
| `timestamp` | ISO 8601 | Log time |
| `upstream_pressure_bar` | bar abs | Mean of ~200 LabJack samples over 20 s window |
| `vacuum_chamber_pressure_mbar` | mbar | Vacuum gauge reading (log-linear conversion) |
| `elflow_mln_min` | mln/min He | Mean of ~20 EL-Flow samples over 20 s window |

## Analysis

```
python GRAPH-PLOT.py                   # opens a file picker
python GRAPH-PLOT.py data/*.csv        # pass files directly
```

Produces two figures (linear and log y-axis) showing:
- Hagen–Poiseuille reference curve (50 µm ID × 2 m, N₂)
- H–P prediction from measured upstream pressures
- EL-Flow measured flow (He-indicated, corrected to actual N₂ via ×0.714)
- Origin-forced linear regression statistics (slope, R², mean ratio Q_meas/Q_HP)

Statistical analysis uses origin-forced linear regression (slope = Σ(Q_HP·Q_meas) / Σ(Q_HP²)) to compare measured flow against the H–P prediction. R² quantifies conformance to the expected P² pressure scaling; the mean ratio Q_meas/Q_HP quantifies the absolute deviation from the H–P prediction. Results and console output are printed per run and summarised across all runs.

## Experiment

Capillary: 50 µm ID × 2 m, 316L stainless steel  
Gas: nitrogen  
Upstream pressure decays from ~1.8 bar to ~1.0 bar absolute over 6–16 hours per run.  
Downstream pressure: ultra-high vacuum (~10⁻⁵ mbar), so ΔP ≈ P_upstream.

## Data

Experimental runs are in the `data/` folder:

| File | Date | Duration |
|---|---|---|
| `gauge_log_20260427_092736.csv` | 27/04/26 | ~12 hours |
| `gauge_log_20260428_090051.csv` | 28/04/26 | ~6 hours |
| `gauge_log_20260428_144730.csv` | 28/04/26 | ~7 hours |
| `gauge_log_20260429_100309.csv` | 29/04/26 | ~16 hours |

Run `gauge_log_20260423_091558.csv` (23/04/26) is excluded from analysis — EL-Flow signal averaging was not yet implemented at the time of collection.
