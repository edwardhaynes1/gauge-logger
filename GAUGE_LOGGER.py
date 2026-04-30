"""
gauge_logger.py  —  LabJack U3 dual-input (FIO0 + FIO2) + EL-Flow data logger
================================================================================
Setup
-----
  LabJack U3  →  4–20 mA pressure transmitter on FIO0 via 100 Ω shunt resistor
  LabJack U3  →  Vacuum gauge analog voltage input on FIO2 (log-linear conversion)
  Serial      →  Bronkhorst F-200CV (propar library, 1 Hz)

Wiring (LabJack U3)
-------------------
  Transmitter +  →  VS  (5 V supply, pin 1)
  Transmitter −  →  FIO0  and one leg of 100 Ω shunt
  Other shunt leg →  GND (pin 2)
  FIO0 reads the voltage across the shunt; I = V / R, P = (I_mA − 4) × (10/16) + 0.13

  Vacuum gauge analog out  →  FIO2 (log-linear: log10(P/mbar) = 5.389 × V − 11.329)

Output
------
  CSV:  timestamp, upstream_pressure_bar, vacuum_chamber_pressure_mbar, elflow_mln_min
  Live table printed to terminal

Requirements
------------
  pip install LabJackPython bronkhorst-propar==0.2.3 pyserial

Configuration
-------------
  Edit the CONFIGURATION block below before first run.
"""

import os
import propar
import u3
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Bronkhorst F-200CV EL-Flow connection
ELFLOW_PORT = None   # None = auto-detect; set to e.g. "COM3" to try that port first
ELFLOW_BAUDRATE = 38400
ELFLOW_NODE = 3         # propar node address (found via scan)
ELFLOW_FULLSCALE = 3.0  # mln/min He

# Logging interval — how often a row is written to CSV
LOG_INTERVAL_S = 20       # seconds between logged rows

# LabJack internal sampling — polls faster than logging to average out quantisation noise
LABJACK_SAMPLE_HZ = 10   # Hz at which the LabJack is actually read
# Each logged upstream pressure value is the mean of all samples in the LOG_INTERVAL_S window
# (e.g. 10 Hz × 20 s = 200 samples averaged per row → noise floor reduced ~14×)

# EL-Flow polling rate — 1 Hz gives ~20 samples per log interval to average
POLLING_HZ = 1            # Hz

# LabJack U3 configuration
# Note: on the U3, FIO pins double as analogue inputs; getAIN(n) reads FIOn.
LABJACK_FIO_UPSTREAM = 0   # FIO pin for upstream transmitter shunt (4–20 mA)
LABJACK_FIO_VACUUM   = 2   # FIO pin for vacuum gauge analog voltage output
LABJACK_SHUNT_OHM    = 100 # shunt resistor in Ω (between FIO0 and GND)

# Vacuum gauge log-linear conversion: log10(P / mbar) = VACUUM_SLOPE × V + VACUUM_INTERCEPT
VACUUM_SLOPE     =  5.389   # decades per volt
VACUUM_INTERCEPT = -11.329  # log10(mbar) at 0 V

# 4–20 mA → pressure conversion parameters
# Formula: pressure_bar = (I_mA − I_MIN) × (P_SPAN / I_SPAN) + P_OFFSET
LABJACK_I_MIN    = 4.0   # mA at zero / minimum pressure
LABJACK_I_SPAN   = 16.0  # mA full-span  (20 − 4)
LABJACK_P_SPAN   = 10.0  # bar full-span of the transmitter
LABJACK_P_OFFSET = 0.13  # bar at I_MIN (zero-pressure offset)

# Output CSV path — saved in the same folder as this script, timestamped at launch
CSV_PATH = Path(__file__).parent / f"gauge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# EL-FLOW AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _probe_elflow_port(port, baudrate, node):
    """Try to open *port* and read from *node*. Returns a propar master on
    success, None on failure. Always closes the connection on failure."""
    try:
        master = propar.master(port, baudrate)
        result = master.read_parameters([{
            "node": node,
            "proc_nr": 33,
            "parm_nr": 0,
            "parm_type": propar.PP_TYPE_FLOAT
        }])
        if result and result[0].get("status") == 0:
            return master
        try:
            master.close()
        except Exception:
            pass
    except Exception:
        pass
    return None


def detect_elflow(hint_port, baudrate, node):
    """Locate the EL-Flow on any available COM port.

    If *hint_port* is set it is tried first; all other ports are then
    scanned in alphabetical order. Raises RuntimeError if nothing responds.
    Returns (port_name, master).
    """
    from serial.tools import list_ports

    candidates = []
    if hint_port:
        candidates.append(hint_port)
    for info in sorted(list_ports.comports(), key=lambda p: p.device):
        if info.device not in candidates:
            candidates.append(info.device)

    if not candidates:
        raise RuntimeError("No COM ports found on this system.")

    print(f"[EL-Flow] Scanning ports: {', '.join(candidates)}")
    for port in candidates:
        print(f"[EL-Flow] Trying {port} …", end=" ", flush=True)
        master = _probe_elflow_port(port, baudrate, node)
        if master is not None:
            print("OK")
            return port, master
        print("no response")

    raise RuntimeError(
        f"EL-Flow not found on any port (baud={baudrate}, node={node}). "
        f"Checked: {', '.join(candidates)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LABJACK U3  — FIO voltage → upstream pressure (4–20 mA) and vacuum pressure
# ─────────────────────────────────────────────────────────────────────────────

def voltage_to_pressure(voltage):
    """Convert FIO0 voltage across shunt resistor to upstream pressure in bar."""
    i_ma = (voltage / LABJACK_SHUNT_OHM) * 1000.0
    if not (LABJACK_I_MIN - 0.5 <= i_ma <= LABJACK_I_MIN + LABJACK_I_SPAN + 0.5):
        return None
    return (i_ma - LABJACK_I_MIN) * (LABJACK_P_SPAN / LABJACK_I_SPAN) + LABJACK_P_OFFSET


def voltage_to_vacuum_pressure(voltage):
    """Convert FIO2 vacuum gauge analog voltage to pressure in mbar (log-linear)."""
    return 10 ** (VACUUM_SLOPE * voltage + VACUUM_INTERCEPT)


def labjack_thread(stop_event):
    sample_interval = 1.0 / LABJACK_SAMPLE_HZ
    while not stop_event.is_set():
        try:
            lj = u3.U3()
            lj.getCalibrationData()
            lj.configIO(FIOAnalog=0x05)   # FIO0 + FIO2 as analogue inputs
            print("[LabJack] Connected and calibrated. FIO0 + FIO2 set to analog.")
        except Exception as e:
            print(f"[LabJack] Could not connect: {e}  — retrying in 5 s")
            time.sleep(5)
            continue

        try:
            while not stop_event.is_set():
                t0 = time.time()
                try:
                    voltage      = lj.getAIN(LABJACK_FIO_UPSTREAM)
                    pressure     = voltage_to_pressure(voltage)
                    vac_voltage  = lj.getAIN(LABJACK_FIO_VACUUM)
                    vac_pressure = voltage_to_vacuum_pressure(vac_voltage)
                except Exception as e:
                    print(f"[LabJack] Read error: {e}")
                    break   # trigger reconnect

                with state["lock"]:
                    if pressure is not None:
                        state["upstream_samples"].append(pressure)
                    state["vacuum_chamber_pressure_mbar"] = vac_pressure

                time.sleep(max(0.0, sample_interval - (time.time() - t0)))
        finally:
            try:
                lj.close()
            except Exception:
                pass
            print("[LabJack] Connection closed — will attempt reconnect.")


# ─────────────────────────────────────────────────────────────────────────────
# EL-FLOW  — Bronkhorst F-200CV via propar library
# ─────────────────────────────────────────────────────────────────────────────

def read_elflow(master):
    if master is None:
        return None
    try:
        result = master.read_parameters([{
            "node": ELFLOW_NODE,
            "proc_nr": 33,
            "parm_nr": 0,
            "parm_type": propar.PP_TYPE_FLOAT
        }])
        if result and result[0].get("status") == 0:
            return round(result[0]["data"], 4)
    except Exception as e:
        print(f"[EL-Flow] Read error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────

state = {
    "upstream_samples": [],    # raw upstream pressure readings, reset each log interval
    "elflow_samples":   [],    # raw EL-Flow readings, reset each log interval
    "vacuum_chamber_pressure_mbar": None,
    "lock": threading.Lock()
}


# ─────────────────────────────────────────────────────────────────────────────
# THREADS
# ─────────────────────────────────────────────────────────────────────────────

def elflow_thread(stop_event):
    interval = 1.0 / POLLING_HZ
    master = None
    while not stop_event.is_set():
        try:
            print("[EL-Flow] Connecting...")
            found_port, master = detect_elflow(ELFLOW_PORT, ELFLOW_BAUDRATE, ELFLOW_NODE)
            print(f"[EL-Flow] Connected on {found_port} node {ELFLOW_NODE}.")
            while not stop_event.is_set():
                t0 = time.time()
                elflow_val = read_elflow(master)
                with state["lock"]:
                    if elflow_val is not None:
                        state["elflow_samples"].append(elflow_val)
                time.sleep(max(0.0, interval - (time.time() - t0)))
        except RuntimeError as e:
            print(f"[EL-Flow] {e}  — retrying in 10 s")
            time.sleep(10)
        except Exception as e:
            import traceback
            print(f"[EL-Flow thread] CRASHED: {e}")
            traceback.print_exc()
            time.sleep(5)
        finally:
            if master is not None:
                try:
                    master.close()
                except Exception:
                    pass
                master = None
            print("[EL-Flow] Connection closed — will attempt reconnect.")


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, decimals=3):
    return f"{val:.{decimals}f}" if val is not None else "---"

def fmt_sci(val, decimals=2):
    return f"{val:.{decimals}e}" if val is not None else "---"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  Gauge Logger  —  press Ctrl-C to stop")
    print("=" * 78)
    print(f"  LabJack U3  : FIO{LABJACK_FIO_UPSTREAM} upstream pressure  {LABJACK_SHUNT_OHM} Ω shunt")
    print(f"  LabJack U3  : FIO{LABJACK_FIO_VACUUM} vacuum gauge  (slope={VACUUM_SLOPE}, intercept={VACUUM_INTERCEPT})")
    port_label = ELFLOW_PORT if ELFLOW_PORT else "auto-detect"
    print(f"  EL-Flow     : {port_label}  {ELFLOW_BAUDRATE} baud  node {ELFLOW_NODE}  (FS={ELFLOW_FULLSCALE} mln/min He)")
    print(f"  Sampling    : LabJack {LABJACK_SAMPLE_HZ} Hz, EL-Flow {POLLING_HZ} Hz  →  averaged & logged every {LOG_INTERVAL_S} s")
    print(f"  Output CSV  : {CSV_PATH}")
    print("=" * 78)

    stop_event = threading.Event()

    t_lj = threading.Thread(target=labjack_thread, args=(stop_event,), daemon=True)
    t_ef = threading.Thread(target=elflow_thread,  args=(stop_event,), daemon=True)

    t_lj.start()
    t_ef.start()

    header = ["timestamp", "upstream_pressure_bar", "vacuum_chamber_pressure_mbar", "elflow_mln_min"]

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        row_count = 0
        print(f"\n{'Timestamp':<26} {'Upstream (bar)':>16} {'Vacuum (mbar)':>16} {'EL-Flow (mln/min)':>18} {'N_P':>6} {'N_EF':>6}")
        print("-" * 92)

        try:
            while True:
                time.sleep(LOG_INTERVAL_S)

                ts = datetime.now().isoformat(timespec="milliseconds")

                with state["lock"]:
                    # Upstream pressure — average and reset
                    up_samp  = state["upstream_samples"]
                    upstream = sum(up_samp) / len(up_samp) if up_samp else None
                    n_up     = len(up_samp)
                    state["upstream_samples"] = []

                    # EL-Flow — average and reset
                    ef_samp = state["elflow_samples"]
                    elflow  = sum(ef_samp) / len(ef_samp) if ef_samp else None
                    n_ef    = len(ef_samp)
                    state["elflow_samples"] = []

                    vacuum = state["vacuum_chamber_pressure_mbar"]

                row = [ts, upstream, vacuum, elflow]
                writer.writerow(row)
                csvfile.flush()
                row_count += 1
                if row_count % 5 == 0:
                    os.fsync(csvfile.fileno())

                print(f"{ts:<26} {fmt(upstream):>16} {fmt_sci(vacuum):>16} {fmt(elflow):>18} {n_up:>6} {n_ef:>6}")

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
        finally:
            stop_event.set()
            t_lj.join(timeout=3)
            t_ef.join(timeout=2)
            print(f"Data saved to {CSV_PATH}  ({row_count} rows)")


if __name__ == "__main__":
    main()