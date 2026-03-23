"""
gauge_logger.py  —  Dual-webcam pressure gauge + EL-Flow data logger
=====================================================================
Setup
-----
  Camera 1  →  analogue needle gauge  (angle detection, ~1 Hz)
  Camera 2  →  digital LCD gauge      (Tesseract OCR, 4 Hz)
  Serial    →  Bronkhorst F-200CV     (propar library, 4 Hz)

Output
------
  CSV:  timestamp, needle_bar, lcd_reading_mbar, elflow_mln_min
  Live table printed to terminal

Requirements
------------
  pip install opencv-python pytesseract bronkhorst-propar==0.2.3
  Tesseract portable:  place tesseract.exe + letsgodigital.traineddata in
    C:\\Users\\uranusprobe\\Documents\\TOF-MS\\capillary flow measurements\\

Configuration
-------------
  Edit the CONFIGURATION block below before first run.
"""

import cv2
import pytesseract
import propar
import os
import csv
import time
import threading
import re
import numpy as np
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# TESSERACT SETUP
# Tesseract is installed in the project folder.
# letsgodigital.traineddata is a model trained on 7-segment display fonts —
# much more accurate than the default eng model for this LCD display.
# ─────────────────────────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\uranusprobe\Documents\TOF-MS\capillary flow measurements\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Users\uranusprobe\Documents\TOF-MS\capillary flow measurements"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Camera indices (0 = laptop built-in, 1 = needle gauge webcam, 2 = LCD webcam)
CAM_NEEDLE = 1   # webcam pointing at analogue needle gauge
CAM_LCD    = 2   # webcam pointing at digital LCD pressure gauge

# Bronkhorst F-200CV EL-Flow connection
ELFLOW_PORT      = "COM3"    # Windows COM port — check Device Manager if this changes
ELFLOW_BAUDRATE  = 38400     # factory default baud rate
ELFLOW_FULLSCALE = 3.0       # mln/min He — from device label (F-200CV-002-AGD-88-V)

# Capture rates
LCD_HZ    = 4    # Hz — LCD OCR and EL-Flow polling rate
NEEDLE_HZ = 1    # Hz — needle gauge polling rate (needle moves slowly)

# Analogue needle gauge calibration
# Measure the needle angle (degrees clockwise from 12 o'clock) at min and max
# scale values and enter below. Current values are estimates — recalibrate for accuracy.
NEEDLE_ANGLE_MIN  = -130   # degrees at 0 bar
NEEDLE_ANGLE_MAX  =  130   # degrees at full scale
NEEDLE_PMIN       =  0.0   # bar at min angle
NEEDLE_PMAX       =  6.0   # bar at max angle

# LCD pressure reading sanity limits — readings outside this range are rejected as misreads
LCD_MBAR_MIN = 1e-8   # mbar — below this is below gauge sensitivity
LCD_MBAR_MAX = 1e3    # mbar — above this is above gauge range

# Output CSV path — saved in the working directory (wherever you run the script from)
CSV_PATH = Path("gauge_log.csv")

# ─────────────────────────────────────────────────────────────────────────────
# NEEDLE GAUGE  — angle → pressure via Hough line detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_needle_angle(frame):
    """
    Detect the needle angle from an analogue gauge camera frame.
    Returns angle in degrees (clockwise from 12 o'clock), or None if detection fails.

    Method:
      1. Convert to greyscale, blur, Canny edge detect
      2. Mask to a circular ROI (inner 80% of frame) to exclude bezel clutter
      3. Hough line transform to find candidate lines
      4. Pick the line whose midpoint is closest to the frame centre (= the needle)
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2   # assumed gauge centre = frame centre

    grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(grey, (5, 5), 0)          # reduce noise before edge detection
    edges = cv2.Canny(blur, 50, 150)                    # detect edges

    # Mask to circular region — ignores edges near the frame border (bezel, reflections)
    mask = np.zeros_like(edges)
    radius = int(min(w, h) * 0.40)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    edges = cv2.bitwise_and(edges, mask)

    # Probabilistic Hough line transform — finds line segments in the edge image
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                             # distance resolution in pixels
        theta=np.pi / 180,                 # angle resolution in radians
        threshold=30,                      # minimum vote count to accept a line
        minLineLength=int(radius * 0.25),  # ignore short lines (noise)
        maxLineGap=10                      # allow small gaps in lines
    )

    if lines is None:
        return None

    # Among all detected lines, pick the one whose midpoint is closest to the gauge centre
    # This selects the needle rather than tick marks or other lines
    best_angle = None
    best_dist  = float("inf")
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dist  = ((mid_x - cx) ** 2 + (mid_y - cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            dx = x2 - x1
            dy = y2 - y1
            # arctan2(x, -y) gives clockwise angle from 12 o'clock
            angle = np.degrees(np.arctan2(dx, -dy))
            best_angle = angle

    return best_angle


def angle_to_pressure(angle):
    """
    Convert needle angle to pressure using linear interpolation between
    the calibration points defined in the CONFIGURATION block.
    Clamps to [NEEDLE_PMIN, NEEDLE_PMAX] to avoid out-of-range values.
    """
    if angle is None:
        return None
    frac = (angle - NEEDLE_ANGLE_MIN) / (NEEDLE_ANGLE_MAX - NEEDLE_ANGLE_MIN)
    frac = max(0.0, min(1.0, frac))   # clamp to [0, 1]
    return NEEDLE_PMIN + frac * (NEEDLE_PMAX - NEEDLE_PMIN)


# ─────────────────────────────────────────────────────────────────────────────
# LCD GAUGE  — Tesseract OCR
# ─────────────────────────────────────────────────────────────────────────────

def _validate_lcd(value):
    """
    Reject OCR results that are physically impossible for this gauge.
    Returns the value unchanged if it's within range, else None.
    """
    if value is None:
        return None
    if LCD_MBAR_MIN <= value <= LCD_MBAR_MAX:
        return value
    return None   # likely a misread


def read_lcd_frame(frame):
    """
    Extract the numeric pressure reading from a single LCD camera frame.
    Returns a float in mbar, or None if parsing fails or value is out of range.

    Processing pipeline:
      1. Extract green channel — LCD digits are bright green on black background
      2. Find bright columns to auto-locate the digit area horizontally
      3. Crop to fixed row bounds (y=125-265) that exclude indicator dots below
      4. Sharpen blurry edges, binarise, upscale 2x for better OCR accuracy
      5. Run Tesseract with letsgodigital 7-segment model
      6. Parse result with three fallback patterns to handle missing dot/E chars

    Row bounds 125-265 are calibrated to the current camera position.
    If the camera is moved, re-run the brightness profile script to recalibrate.
    """
    # ── Step 1: locate digit columns ─────────────────────────────────────────
    g = cv2.split(frame)[1]           # green channel
    col_max = g.max(axis=0)           # max brightness per column
    max_brightness = g.max()          # brightest pixel in the frame
    threshold = max_brightness * 0.6  # columns above 60% of peak = digit columns

    bright_cols = [x for x in range(frame.shape[1]) if col_max[x] > threshold]
    if not bright_cols:
        return None   # display may be off or camera not pointed at gauge

    x1, x2 = bright_cols[0], bright_cols[-1]

    # ── Step 2: crop to main digit row only ──────────────────────────────────
    # y=125-265 = main digits; y=300-365 = small indicator LEDs (excluded)
    roi = frame[125:265, x1:x2]

    # ── Step 3: preprocess ───────────────────────────────────────────────────
    g_roi   = cv2.split(roi)[1]                               # green channel of ROI
    blurred = cv2.GaussianBlur(g_roi, (0, 0), 2)             # smooth noise
    sharp   = cv2.addWeighted(g_roi, 2.0, blurred, -1.0, 0)  # unsharp mask to recover blurry edges
    _, thresh = cv2.threshold(sharp, 80, 255, cv2.THRESH_BINARY)  # binarise
    thresh  = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # 2x upscale

    # ── Step 4: OCR ──────────────────────────────────────────────────────────
    # PSM 7  = treat image as a single text line
    # OEM 0  = legacy Tesseract engine (required by letsgodigital model)
    # letsgodigital = trained on 7-segment display fonts
    config = "--psm 7 --oem 0 -l letsgodigital -c tessedit_char_whitelist=0123456789.-Ee+"
    text   = pytesseract.image_to_string(thresh, config=config).strip()

    # ── Step 5: parse OCR result ─────────────────────────────────────────────
    # Pattern 1: Tesseract read the full value correctly e.g. "4.7E-3"
    m = re.search(r"(\d+\.?\d*)[Ee]([+-]?\d+)", text)
    if m:
        return _validate_lcd(float(f"{m.group(1)}e{m.group(2)}"))

    # Pattern 1b: decimal point present but E missing e.g. "4.5-3" → 4.5e-3
    # Happens when the display's E character blurs into nothing
    m = re.search(r"(\d+\.\d+)-(\d+)", text)
    if m:
        return _validate_lcd(float(f"{m.group(1)}e-{m.group(2)}"))

    # Pattern 2: both decimal point and E missing e.g. "47-3" → 4.7e-3
    # Most common case — blurry decimal point and E both lost
    m = re.search(r"(\d)(\d+)-(\d+)", text)
    if m:
        return _validate_lcd(float(f"{m.group(1)}.{m.group(2)}e-{m.group(3)}"))

    # Pattern 3: plain number fallback e.g. if exponent not visible at all
    m = re.search(r"[-+]?\d+\.?\d*", text)
    if m:
        try:
            return _validate_lcd(float(m.group()))
        except ValueError:
            pass

    return None   # all patterns failed


# ─────────────────────────────────────────────────────────────────────────────
# EL-FLOW  — Bronkhorst F-200CV via propar library
# ─────────────────────────────────────────────────────────────────────────────

def open_elflow():
    """
    Connect to the EL-Flow using Bronkhorst's propar library.
    propar.instrument signature: (comport, node_address, baudrate)
    Note: node_address=3 is the Bronkhorst factory default.
    Returns a propar instrument object, or None on failure.
    """
    try:
        instrument = propar.instrument(ELFLOW_PORT, 3, ELFLOW_BAUDRATE)
        print(f"[EL-Flow] Connected on {ELFLOW_PORT}")
        return instrument
    except Exception as e:
        print(f"[EL-Flow] Could not connect on {ELFLOW_PORT}: {e}")
        return None


def read_elflow(instrument):
    """
    Read the current flow measurement from the EL-Flow.
    Parameter 205 returns the live flow value directly in mln/min.
    (Parameter 33 = raw 0-32000 counts, parameter 206 = full scale — not used here.)
    Returns flow in mln/min rounded to 4 decimal places, or None on failure.
    """
    if instrument is None:
        return None
    try:
        val = instrument.readParameter(205)   # live flow in mln/min
        if val is not None:
            return round(val, 4)
    except Exception as e:
        print(f"[EL-Flow] Read error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# Threads write sensor readings here; the main loop reads and logs them.
# A threading.Lock prevents simultaneous read/write corruption.
# ─────────────────────────────────────────────────────────────────────────────

state = {
    "needle_bar":     None,   # analogue gauge pressure in bar
    "lcd_value":      None,   # LCD gauge pressure in mbar
    "elflow_mln_min": None,   # EL-Flow flow rate in mln/min
    "lock":           threading.Lock()
}


# ─────────────────────────────────────────────────────────────────────────────
# CAPTURE THREADS
# Each sensor runs in its own daemon thread so a slow/failing sensor
# cannot block the others. Daemon threads are killed automatically when
# the main process exits.
# ─────────────────────────────────────────────────────────────────────────────

def needle_thread(stop_event):
    """Captures frames from the needle gauge camera and updates state['needle_bar']."""
    cap = cv2.VideoCapture(CAM_NEEDLE)
    if not cap.isOpened():
        print(f"[Needle cam] Could not open camera index {CAM_NEEDLE}")
        return
    interval = 1.0 / NEEDLE_HZ
    while not stop_event.is_set():
        t0 = time.time()
        ret, frame = cap.read()
        if ret:
            angle    = detect_needle_angle(frame)
            pressure = angle_to_pressure(angle)
            with state["lock"]:
                state["needle_bar"] = pressure
        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))   # sleep only the remaining time
    cap.release()


def lcd_thread(stop_event):
    """
    Captures frames from the LCD gauge camera and updates state['lcd_value'].
    Uses DirectShow backend and fixed exposure to prevent auto-exposure drift
    which would throw off the brightness-based digit detection.
    """
    try:
        cap = cv2.VideoCapture(CAM_LCD, cv2.CAP_DSHOW)   # DirectShow — more reliable on Windows
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)            # disable auto-exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)                # fixed exposure tuned for this display
        time.sleep(0.5)                                   # wait for exposure to settle
        if not cap.isOpened():
            print(f"[LCD cam] Could not open camera index {CAM_LCD} — LCD logging disabled")
            return
        print(f"[LCD cam] Camera {CAM_LCD} opened")
        interval = 1.0 / LCD_HZ
        while not stop_event.is_set():
            t0 = time.time()
            ret, frame = cap.read()
            if ret:
                lcd_val = read_lcd_frame(frame)
                with state["lock"]:
                    state["lcd_value"] = lcd_val
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))
        cap.release()
    except Exception as e:
        import traceback
        print(f"[LCD thread] CRASHED: {e}")
        traceback.print_exc()


def elflow_thread(stop_event):
    """
    Connects to the EL-Flow and continuously polls flow rate into state['elflow_mln_min'].
    The instrument is created inside the thread — propar is not thread-safe so the
    instrument object must not be shared across threads.
    """
    try:
        print("[EL-Flow thread] connecting...")
        instrument = propar.instrument(ELFLOW_PORT, 3, ELFLOW_BAUDRATE)
        print("[EL-Flow thread] connected, entering loop")
        interval = 1.0 / LCD_HZ   # poll at same rate as LCD
        while not stop_event.is_set():
            t0 = time.time()
            elflow_val = read_elflow(instrument)
            with state["lock"]:
                state["elflow_mln_min"] = elflow_val
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))
    except Exception as e:
        import traceback
        print(f"[EL-Flow thread] CRASHED: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, decimals=3):
    """Format a float to fixed decimal places, or '---' if None."""
    return f"{val:.{decimals}f}" if val is not None else "---"

def fmt_sci(val):
    """Format LCD pressure in scientific notation e.g. 4.7e-03, or '---' if None."""
    return f"{val:.1e}" if val is not None else "---"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP  — logging and display
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  Gauge Logger  —  press Ctrl-C to stop")
    print("=" * 66)
    print(f"  Needle cam  : index {CAM_NEEDLE}   ({NEEDLE_HZ} Hz)")
    print(f"  LCD cam     : index {CAM_LCD}   ({LCD_HZ} Hz)")
    print(f"  EL-Flow     : {ELFLOW_PORT}  {ELFLOW_BAUDRATE} baud  (FS={ELFLOW_FULLSCALE} mln/min He)")
    print(f"  Output CSV  : {CSV_PATH}")
    print("=" * 66)

    # Start all three sensor threads as daemons
    stop_event = threading.Event()
    t1 = threading.Thread(target=needle_thread, args=(stop_event,), daemon=True)
    t2 = threading.Thread(target=lcd_thread,    args=(stop_event,), daemon=True)
    t3 = threading.Thread(target=elflow_thread, args=(stop_event,), daemon=True)
    t1.start()
    t2.start()
    t3.start()

    time.sleep(1.5)   # give cameras and EL-Flow time to initialise before logging starts

    header       = ["timestamp", "needle_bar", "lcd_reading_mbar", "elflow_mln_min"]
    log_interval = 1.0 / LCD_HZ   # log at the fastest sensor rate

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        row_count = 0
        print(f"\n{'Timestamp':<26} {'Needle (bar)':>14} {'LCD (mbar)':>12} {'EL-Flow (mln/min)':>18}")
        print("-" * 74)

        try:
            while True:
                t0 = time.time()
                ts = datetime.now().isoformat(timespec="milliseconds")

                # Read latest values from shared state
                with state["lock"]:
                    needle = state["needle_bar"]
                    lcd    = state["lcd_value"]
                    elflow = state["elflow_mln_min"]

                # Write to CSV on every loop iteration (at LCD_HZ rate)
                row = [ts, needle, lcd, elflow]
                writer.writerow(row)
                csvfile.flush()   # flush so data is visible in the file during a run

                # Print to terminal at half the log rate to keep output readable
                row_count += 1
                if row_count % max(1, LCD_HZ // 2) == 0:
                    print(f"{ts:<26} {fmt(needle):>14} {fmt_sci(lcd):>12} {fmt(elflow):>18}")

                elapsed = time.time() - t0
                time.sleep(max(0.0, log_interval - elapsed))

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
        finally:
            # Signal all threads to stop and wait for them to finish cleanly
            stop_event.set()
            t1.join(timeout=2)
            t2.join(timeout=2)
            t3.join(timeout=2)
            print(f"Data saved to {CSV_PATH}  ({row_count} rows)")


if __name__ == "__main__":
    main()