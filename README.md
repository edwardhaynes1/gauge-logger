# gauge-logger

Data acquisition system for capillary flow characterisation. Logs upstream pressure, vacuum chamber pressure, and mass flow rate to CSV for Hagen–Poiseuille validation experiments.

## Hardware

| Component | Description |
|-----------|-------------|
| LabJack U3 | DAQ device — analog inputs for pressure sensors |
| GEMS 2200 Series | Upstream pressure transmitter, 0–10 bar absolute, 4–20 mA, via 98.6 Ω shunt on FIO0 |
| Vacuum gauge | Log-linear analog output on FIO2 |
| Bronkhorst F-200CV EL-Flow | Mass flow meter (He-calibrated, 3.0 mln/min FS) |
| Keller Series 30 | Absolute pressure/temperature reference (calibration runs only) |

### Wiring (LabJack U3)

```
Transmitter +      →  VS (5 V, pin 1)
Transmitter −      →  FIO0 + one leg of 98.6 Ω shunt
Shunt other leg    →  GND (pin 2)
Vacuum gauge out   →  FIO2
```

## Software dependencies

```bash
pip install LabJackPython bronkhorst-propar==0.2.3 pyserial
```

For calibration runs with the Keller attached:
```bash
pip install keller-protocol
```

For the analysis plotter:
```bash
pip install matplotlib numpy scipy
```

## Configuration

Edit the CONFIGURATION block at the top of `GAUGE_LOGGER.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| ELFLOW_PORT | None | COM port for EL-Flow. None = auto-detect |
| ELFLOW_NODE | 3 | Propar node address |
| ELFLOW_FULLSCALE | 3.0 | Full-scale flow in mln/min He |
| KELLER_ENABLED | False | Set True to enable Keller logging (calibration runs only) |
| KELLER_PORT | None | COM port for Keller. None = auto-detect |
| LOG_INTERVAL_S | 20 | Seconds between logged rows |
| LABJACK_SAMPLE_HZ | 10 | LabJack internal sampling rate (Hz) |
| LABJACK_SHUNT_OHM | 98.6 | Measured shunt resistance (Ω) |
| LABJACK_P_OFFSET | 0.036 | Zero offset (bar) — calibrated to MeteoSwiss BER 953 hPa |
| LABJACK_C_FACTOR | 1.0201 | Gain correction factor — from Keller cross-calibration 29/05/2026 |
| VACUUM_SLOPE | 5.389 | Log-linear vacuum gauge calibration slope |
| VACUUM_INTERCEPT | -11.329 | Log-linear vacuum gauge calibration intercept |

## Calibration

Full calibration history is documented in `CALIBRATION/GEMS_Calibration_Record.docx`.

### Event 1 — 15/05/2026 (shunt and offset correction)

Applied in `GAUGE_LOGGER.py` from 15/05/2026 onwards.

- Shunt resistance corrected: 100.0 Ω (assumed) → 98.6 Ω (measured with calibrated multimeter)
- Atmospheric reference corrected: 1.000 bar (assumed) → 0.953 bar (MeteoSwiss BER, 20/03/2026)
- Constants updated: `LABJACK_SHUNT_OHM = 98.6`, `LABJACK_P_OFFSET = 0.036`

Data logged before 15/05/2026 requires post-acquisition correction via `CORRECT_GAUGE-LOGS.py` (see Data correction section).

**Correction formula (historical data only):**
```
P_corrected = ((((P_logged - 0.13) / 0.625) + 4) × 1.0142 - 4) × 0.625 + 0.036
```

### Event 2 — 29/05/2026 (Keller cross-calibration, gain correction)

Applied in `GAUGE_LOGGER.py` from 29/05/2026 onwards.

- Keller Series 30 absolute transmitter used as independent reference across 0.93–3.60 bar (N₂, capillary bench)
- OLS regression (Keller as reference): `Keller = 1.0201 × GEMS + 0.0358`
- GEMS reads ~2.0% low across the measured range — gain correction applied: `LABJACK_C_FACTOR = 1.0201`
- Intercept (0.0358 bar) consistent with existing `LABJACK_P_OFFSET` — no separate change needed

Data logged between 15/05/2026 and 29/05/2026 is offset-corrected but carries an additional ~2% systematic uncertainty in upstream pressure. This gain correction has not been applied retroactively; the gain error cannot be verified to have been constant throughout the earlier logging period.

**Calibration traceability:** factory (Keller) → Keller Series 30 → GEMS 2200 Series.

## Usage

```bash
python GAUGE_LOGGER.py
```

The EL-Flow and Keller (if enabled) are auto-detected on startup by scanning all available COM ports. The LabJack U3 is detected automatically by device type.

Output is written to a timestamped CSV in the same folder as the script:
```
gauge_log_YYYYMMDD_HHMMSS.csv
```

### Running a calibration cross-check

To log GEMS and Keller simultaneously for cross-calibration validation:

1. Attach the Keller to the upstream line (tee fitting alongside the GEMS)
2. Set `KELLER_ENABLED = True` in the configuration block
3. Run `GAUGE_LOGGER.py` as normal
4. Vary upstream pressure across the full range with deliberate dwell periods at stable setpoints
5. Set `KELLER_ENABLED = False` when done

## Data correction

**Only applies to data collected before 15/05/2026**

Historical raw CSV files (in `ARCHIVED-DATA/UNCALIBRATED-SHUNT-DATA/`) contain uncorrected pressure readings.

```bash
python CORRECT_GAUGE-LOGS.py
```

Select the file to correct. Output saved as `*_CORRECTED.csv`. Original file is never modified.

Note: `CORRECT_GAUGE-LOGS.py` applies Event 1 corrections only. The ~2% gain correction from Event 2 is not applied retroactively — see calibration section above.

The original version of this script used on 15/05/2026 to generate the `_CORRECTED.csv` files is preserved at `ARCHIVED-DATA/UNCALIBRATED-SHUNT-DATA/CORRECT_GAUGE-LOGS_v1_15052026.py` for full reproducibility.

## CSV columns

| Column | Units | Description |
|--------|-------|-------------|
| timestamp | ISO 8601 | Log time |
| upstream_pressure_bar | bar abs | Mean of ~200 LabJack samples over 20 s window |
| vacuum_chamber_pressure_mbar | mbar | Vacuum gauge reading (log-linear conversion) |
| elflow_mln_min | mln/min He | Mean of ~20 EL-Flow samples over 20 s window |
| keller_p1_bar | bar abs | Keller P1 channel — present only in calibration runs (`KELLER_ENABLED = True`), otherwise empty |
| keller_tob1_degC | °C | Keller temperature channel — calibration runs only, otherwise empty |

## Analysis

```bash
python GRAPH-PLOT.py              # opens a file picker
python GRAPH-PLOT.py data/*.csv   # pass files directly
```

Produces two figures (linear and log y-axis) showing:
- Hagen–Poiseuille reference curve (50 µm ID × 2 m, N₂)
- H–P prediction from measured upstream pressures
- EL-Flow measured flow (He-indicated, corrected to actual N₂ via ×0.714)
- Origin-forced linear regression statistics (slope, R², mean ratio Q_meas/Q_HP)

Statistical analysis uses origin-forced linear regression (slope = Σ(Q_HP·Q_meas) / Σ(Q_HP²)) to compare measured flow against the H–P prediction. R² quantifies conformance to the expected P² pressure scaling; the mean ratio Q_meas/Q_HP quantifies the absolute deviation from the H–P prediction.

## Experiment

- **Capillary:** 50 µm ID × 2 m, 316L stainless steel
- **Gas:** nitrogen
- **Upstream pressure:** decays from ~1.8 bar to ~1.0 bar absolute over 6–24 hours per run
- **Downstream pressure:** high vacuum (~10⁻⁷ mbar), so ΔP ≈ P_upstream

## Data

Corrected experimental runs are in the `data/` folder:

| File | Date | Duration |
|------|------|----------|
| gauge_log_20260427_092736_CORRECTED.csv | 27/04/26 | ~12 hours |
| gauge_log_20260428_090051_CORRECTED.csv | 28/04/26 | ~6 hours |
| gauge_log_20260428_144730_CORRECTED.csv | 28/04/26 | ~7 hours |
| gauge_log_20260429_100309_CORRECTED.csv | 29/04/26 | ~16 hours |
| gauge_log_20260430_082858_CORRECTED.csv | 30/04/26 | ~24 hours |

Uncorrected raw data files are archived in `ARCHIVED-DATA/UNCALIBRATED-SHUNT-DATA/`.

`gauge_log_20260423_091558.csv` (23/04/26) is excluded from analysis — EL-Flow signal averaging was not yet implemented at the time of collection.

Atmospheric pressure reference data from MeteoSwiss Bern station (20/03/2026) is archived in `ARCHIVED-DATA/METAS-PRESSURE-REFERENCE/`.

---

## Repository structure

```
CAPILLARY-GAUGE-LOGGER/
│
├── GAUGE_LOGGER.py                        # Main data acquisition script (current version)
├── CORRECT_GAUGE-LOGS.py                  # Post-acquisition correction for pre-15/05/2026 data
├── GRAPH-PLOT.py                          # Hagen–Poiseuille analysis plotter
├── GRAPH-PLOT-PRESENTATION.py             # Presentation-formatted version of plotter
├── README.md                              # This file
│
├── CALIBRATION/                           # Calibration documentation and raw calibration data
│   └── GEMS-AND-KELLER-LOGGING/           # Calibration Event 2 (29/05/2026) — Keller cross-comparison
│       ├── LOGGING-ONLY-GEMS-AND-KELLER/  # Earlier GEMS+Keller logging scripts (pre-merge)
│       ├── gauge_log_20260529_131841.csv  # Raw calibration session data
│       ├── gauge_log_20260529_151318.csv
│       ├── gauge_log_20260529_160713.csv
│       ├── GAUGE_LOGGER.py                # Logger version used during calibration session
│       └── GEMS_Calibration_Record.docx   # Full calibration record (Events 1 and 2)
│
└── ARCHIVED-DATA/                         # Historical data — not used in current analysis
    ├── UNCALIBRATED-SHUNT-DATA/           # Raw data before 15/05/2026 (uncorrected shunt/offset)
    │   └── CORRECT_GAUGE-LOGS_v1_15052026.py  # Original correction script used to generate _CORRECTED files
    ├── PRESENTATION-DATA-260527/          # Data used for 27/05/2026 statistics presentation
    ├── METAS-PRESSURE-REFERENCE/          # MeteoSwiss BER atmospheric pressure reference (20/03/2026)
    └── LOG_ARCHIVE/                       # Miscellaneous early log files
```

**Active experimental data** (corrected runs used in Hagen–Poiseuille analysis) lives alongside the scripts in the root folder, named `gauge_log_YYYYMMDD_HHMMSS_CORRECTED.csv`.
