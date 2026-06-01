"""
GAUGE_LOGGER_KELLER_GEMS.py
===========================
Logs only:
  - GEMS upstream pressure via LabJack U3 on FIO0 through a shunt resistor
  - KELLER pressure/temperature via official keller-protocol package on COM15

Requirements
------------
  pip install LabJackPython pyserial keller-protocol

Output CSV columns
------------------
  timestamp, gems_upstream_pressure_bar, keller_p1_bar, keller_tob1_degC
"""

import os
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

import u3
from keller_protocol import keller_protocol as kp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LOG_INTERVAL_S = 2
LABJACK_SAMPLE_HZ = 10

LABJACK_FIO_UPSTREAM = 0
LABJACK_SHUNT_OHM = 98.6
LABJACK_P_OFFSET = 0.036
LABJACK_I_MIN = 4.0
LABJACK_I_SPAN = 16.0
LABJACK_P_SPAN = 10.0

KELLER_PORT = 'COM15'
KELLER_BAUD = 9600
KELLER_ADDR = 250
KELLER_TIMEOUT = 0.3
KELLER_ECHO = True
KELLER_POLL_HZ = 1

CSV_PATH = Path(__file__).parent / f"gauge_log_keller_gems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSIONS
# ─────────────────────────────────────────────────────────────────────────────
def voltage_to_pressure(voltage):
    i_ma = (voltage / LABJACK_SHUNT_OHM) * 1000.0
    if not (LABJACK_I_MIN - 0.5 <= i_ma <= LABJACK_I_MIN + LABJACK_I_SPAN + 0.5):
        return None
    return (i_ma - LABJACK_I_MIN) * (LABJACK_P_SPAN / LABJACK_I_SPAN) + LABJACK_P_OFFSET


def fmt(val, decimals=3):
    return f"{val:.{decimals}f}" if val is not None else "---"


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────
state = {
    'gems_upstream_samples': [],
    'keller_p1': None,
    'keller_tob1': None,
    'lock': threading.Lock(),
}


# ─────────────────────────────────────────────────────────────────────────────
# LABJACK THREAD
# ─────────────────────────────────────────────────────────────────────────────
def labjack_thread(stop_event):
    sample_interval = 1.0 / LABJACK_SAMPLE_HZ
    while not stop_event.is_set():
        try:
            lj = u3.U3()
            lj.getCalibrationData()
            lj.configIO(FIOAnalog=0x01)
            print(f"[LabJack] Connected and calibrated. FIO{LABJACK_FIO_UPSTREAM} set to analog.")
        except Exception as e:
            print(f"[LabJack] Could not connect: {e} — retrying in 5 s")
            time.sleep(5)
            continue

        try:
            while not stop_event.is_set():
                t0 = time.time()
                try:
                    voltage = lj.getAIN(LABJACK_FIO_UPSTREAM)
                    pressure = voltage_to_pressure(voltage)
                except Exception as e:
                    print(f"[LabJack] Read error: {e}")
                    break

                with state['lock']:
                    if pressure is not None:
                        state['gems_upstream_samples'].append(pressure)

                time.sleep(max(0.0, sample_interval - (time.time() - t0)))
        finally:
            try:
                lj.close()
            except Exception:
                pass
            print('[LabJack] Connection closed — will attempt reconnect.')


# ─────────────────────────────────────────────────────────────────────────────
# KELLER THREAD
# ─────────────────────────────────────────────────────────────────────────────
def keller_thread(stop_event):
    interval = 1.0 / KELLER_POLL_HZ
    bus = None
    while not stop_event.is_set():
        try:
            print(f"[Keller] Connecting on {KELLER_PORT} at {KELLER_BAUD} baud ...")
            bus = kp.KellerProtocol(
                port=KELLER_PORT,
                baud_rate=KELLER_BAUD,
                timeout=KELLER_TIMEOUT,
                echo=KELLER_ECHO,
            )
            fw = bus.f48(KELLER_ADDR)
            print(f"[Keller] Init OK: {fw}")

            while not stop_event.is_set():
                t0 = time.time()
                try:
                    p1 = bus.f73(KELLER_ADDR, 1)
                    tob1 = bus.f73(KELLER_ADDR, 4)
                    with state['lock']:
                        state['keller_p1'] = p1
                        state['keller_tob1'] = tob1
                except Exception as e:
                    print(f"[Keller] Read error: {e}")
                    break
                time.sleep(max(0.0, interval - (time.time() - t0)))
        except Exception as e:
            print(f"[Keller] Could not connect/init: {e} — retrying in 5 s")
            time.sleep(5)
        finally:
            bus = None
            print('[Keller] Connection closed — will attempt reconnect.')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 86)
    print('  Gauge Logger — KELLER + GEMS upstream only  —  press Ctrl-C to stop')
    print('=' * 86)
    print(f'  LabJack U3  : FIO{LABJACK_FIO_UPSTREAM} GEMS upstream pressure  {LABJACK_SHUNT_OHM} Ω shunt')
    print(f'  Keller      : {KELLER_PORT}  {KELLER_BAUD} baud  address {KELLER_ADDR}')
    print(f'  Sampling    : LabJack {LABJACK_SAMPLE_HZ} Hz, Keller {KELLER_POLL_HZ} Hz  →  logged every {LOG_INTERVAL_S} s')
    print(f'  Output CSV  : {CSV_PATH}')
    print('=' * 86)

    stop_event = threading.Event()
    t_lj = threading.Thread(target=labjack_thread, args=(stop_event,), daemon=True)
    t_ke = threading.Thread(target=keller_thread, args=(stop_event,), daemon=True)
    t_lj.start()
    t_ke.start()

    header = ['timestamp', 'gems_upstream_pressure_bar', 'keller_p1_bar', 'keller_tob1_degC']

    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        row_count = 0
        print(f"\n{'Timestamp':<26} {'GEMS up (bar)':>16} {'Keller P1 (bar)':>18} {'Keller TOB1 (°C)':>18} {'N_GEMS':>8}")
        print('-' * 92)

        try:
            while True:
                time.sleep(LOG_INTERVAL_S)
                ts = datetime.now().isoformat(timespec='milliseconds')

                with state['lock']:
                    gems_samp = state['gems_upstream_samples']
                    gems_up = sum(gems_samp) / len(gems_samp) if gems_samp else None
                    n_gems = len(gems_samp)
                    state['gems_upstream_samples'] = []

                    keller_p1 = state['keller_p1']
                    keller_tob1 = state['keller_tob1']

                row = [ts, gems_up, keller_p1, keller_tob1]
                writer.writerow(row)
                csvfile.flush()
                row_count += 1
                if row_count % 5 == 0:
                    os.fsync(csvfile.fileno())

                print(f"{ts:<26} {fmt(gems_up):>16} {fmt(keller_p1):>18} {fmt(keller_tob1, 2):>18} {n_gems:>8}")

        except KeyboardInterrupt:
            print('\n\nStopped by user.')
        finally:
            stop_event.set()
            t_lj.join(timeout=3)
            t_ke.join(timeout=3)
            print(f'Data saved to {CSV_PATH}  ({row_count} rows)')


if __name__ == '__main__':
    main()