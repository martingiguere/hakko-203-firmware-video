#!/usr/bin/env python3
"""
Calibrate the character grid for the Xeltek SuperPro 6100N "Edit Buffer" hex dump.

Uses a video frame (or the reference screenshot) to determine: character cell size,
grid origin, row/column layout, byte group spacing, and dash position.

Key differences from FM-202 (Segger Flasher):
  - 10-digit addresses (not 6)
  - 8-dash-8 byte grouping with non-uniform spacing
  - Green column header bars (ADDRESS / HEX / ASCII) to detect and skip
  - Color image needed for green header detection
"""

import cv2
import numpy as np
import json
import sys
import os


# Default calibration frame — override with command-line argument
DEFAULT_CALIBRATION_FRAME = 'frames/frame_00100.png'


# --- Green header detection (Xeltek-specific) ---

def detect_green_headers(img_color):
    """
    Detect the green column header bars (ADDRESS, HEX, ASCII) using HSV
    color thresholding. Returns the bounding box of the green header region,
    which constrains both the y-start and x-extent of the hex data area.

    Returns:
        (header_bottom_y, x_left, x_right): y below headers, and x-extent
        of the dialog. Returns (0, 0, img_width) if no headers detected.
    """
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    h, w = img_color.shape[:2]

    # Green hue range: 35-85, high saturation (>80), medium-high value (>80)
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find rows with significant green content
    row_green = np.sum(mask > 0, axis=1)
    green_threshold = w * 0.05  # at least 5% of row width is green

    green_rows = np.where(row_green > green_threshold)[0]
    if len(green_rows) == 0:
        print("  Warning: no green header bars detected")
        return 0, 0, w

    header_bottom_y = int(green_rows[-1]) + 1
    header_height = green_rows[-1] - green_rows[0] + 1

    # Use the green bars' x-extent to find the dialog bounds and column positions.
    # Require each column to be green across >50% of header height to filter noise.
    header_mask = mask[green_rows[0]:green_rows[-1] + 1, :]
    col_green = np.sum(header_mask > 0, axis=0)
    min_green_rows = max(header_height * 0.15, 3)
    green_cols = np.where(col_green > min_green_rows)[0]

    if len(green_cols) < 10:
        print("  Warning: couldn't detect green header x-extent")
        return header_bottom_y, 0, w, []

    x_left = int(green_cols[0])
    x_right = int(green_cols[-1])

    # Find the separate green bars by detecting gaps in the green column profile.
    # Use a larger gap threshold to avoid splitting bars due to camera artifacts.
    bars = []
    bar_start = green_cols[0]
    for i in range(1, len(green_cols)):
        if green_cols[i] - green_cols[i - 1] > 20:  # gap > 20px = new bar
            bars.append((int(bar_start), int(green_cols[i - 1])))
            bar_start = green_cols[i]
    bars.append((int(bar_start), int(green_cols[-1])))

    print(f"  Green headers detected: rows {green_rows[0]}–{green_rows[-1]}, "
          f"x=[{x_left}, {x_right}]")
    if len(bars) >= 3:
        print(f"  Header bars: ADDRESS=[{bars[0][0]},{bars[0][1]}], "
              f"HEX=[{bars[1][0]},{bars[1][1]}], "
              f"ASCII=[{bars[2][0]},{bars[2][1]}]")
    elif len(bars) >= 1:
        print(f"  Header bars ({len(bars)} detected): "
              f"{', '.join(f'[{b[0]},{b[1]}]' for b in bars)}")
    print(f"  Data region starts at y={header_bottom_y}")
    return header_bottom_y, x_left, x_right, bars


# --- Hex dump region detection ---

def find_hex_dump_region(img_gray, img_color=None):
    """
    Find the bounding box of the hex dump text area in the frame.

    For video frames: uses green header detection (needs color image) to find
    the top of the data region and x-extent of the dialog, then row-variance
    analysis to refine vertical bounds.

    For the reference screenshot: the Edit Buffer dialog occupies most of the image.
    """
    h, w = img_gray.shape

    # If we have a color image, try green header detection first
    header_bottom_y = 0
    dialog_x_left = 0
    dialog_x_right = w
    green_bars = []
    if img_color is not None:
        header_bottom_y, dialog_x_left, dialog_x_right, green_bars = \
            detect_green_headers(img_color)

    # Search region: below green headers, constrained to dialog x-bounds
    search_top = max(header_bottom_y, int(h * 0.05))
    search_bottom = int(h * 0.75)
    search_left = dialog_x_left
    search_right = dialog_x_right

    search_region = img_gray[search_top:search_bottom, search_left:search_right]

    # Row variance analysis: text rows have high variance (dark chars on light bg)
    row_var = np.var(search_region.astype(float), axis=1)
    threshold = np.mean(row_var) * 0.5
    text_rows = row_var > threshold

    text_indices = np.where(text_rows)[0]
    if len(text_indices) < 10:
        print("  Warning: couldn't reliably detect text region via variance")
        return search_left, search_top, search_right, search_bottom

    y_start = search_top + text_indices[0]
    y_end = search_top + text_indices[-1]

    # x bounds come from the green header extent (dialog_x_left/right)
    # Refine with column variance within the data region
    text_region = img_gray[y_start:y_end, search_left:search_right]
    col_var = np.var(text_region.astype(float), axis=0)
    col_threshold = np.mean(col_var) * 0.3
    text_cols = col_var > col_threshold
    text_col_indices = np.where(text_cols)[0]

    if len(text_col_indices) < 10:
        x_start = search_left
        x_end = search_right
    else:
        x_start = search_left + int(text_col_indices[0])
        x_end = search_left + int(text_col_indices[-1])

    return x_start, y_start, x_end, y_end


# --- Row height measurement ---

def measure_row_height(img, y_start, y_end):
    """
    Measure row height using autocorrelation of vertical intensity profile.
    More robust than transition detection since it finds the dominant period
    directly, regardless of noise or partial rows.
    Returns (row_height, None).
    """
    h, w = img.shape
    # Average across the middle portion of the frame horizontally
    # to get a robust vertical intensity profile
    strip_x0 = int(w * 0.25)
    strip_x1 = int(w * 0.60)
    strip = img[y_start:y_end, strip_x0:strip_x1]

    row_means = np.mean(strip.astype(float), axis=1)
    centered = row_means - np.mean(row_means)

    if len(centered) < 40:
        return 25, None  # fallback for very short regions

    # Autocorrelation
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
    if autocorr[0] > 0:
        autocorr /= autocorr[0]

    # Find first peak in the expected range [15, 40] pixels
    # (typical for hex dump text rows at 1080p)
    min_lag = 15
    max_lag = min(40, len(autocorr) - 1)
    if max_lag <= min_lag:
        return 25, None

    search = autocorr[min_lag:max_lag + 1]
    peak_idx = np.argmax(search) + min_lag
    row_height = int(round(peak_idx))

    # Cross-check with FFT
    fft_height, _ = estimate_row_height_fft(row_means)

    print(f"  Autocorrelation row height: {row_height}px (corr={autocorr[peak_idx]:.3f})")
    print(f"  FFT row height: {fft_height}px")

    # If they agree (within 3px), use autocorrelation result
    # If they disagree, prefer the one closer to 25 (expected for 1080p Xeltek)
    if abs(row_height - fft_height) <= 3:
        print(f"  Both methods agree → using {row_height}px")
    else:
        expected = 25
        if abs(fft_height - expected) < abs(row_height - expected):
            row_height = fft_height
            print(f"  Methods disagree; FFT closer to expected → using {row_height}px")
        else:
            print(f"  Methods disagree; autocorr closer to expected → using {row_height}px")

    return row_height, None


def estimate_row_height_fft(row_means):
    """Estimate row height using FFT of vertical intensity profile."""
    centered = row_means - np.mean(row_means)
    fft = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered))

    # Ignore DC and very low frequencies
    fft[:3] = 0

    peak_idx = np.argmax(fft)
    peak_freq = freqs[peak_idx]

    if peak_freq > 0:
        period = int(round(1.0 / peak_freq))
        print(f"  FFT-estimated row height: {period} pixels (freq={peak_freq:.4f})")
        return period, None

    return 20, None  # fallback


# --- Character width measurement ---

def measure_char_width(img, y_start, row_height, x_start, x_end):
    """
    Measure character width using FFT of horizontal intensity profile
    within a single row of hex text.

    The FFT may pick up the 3-char byte-pair period (2 hex digits + space)
    rather than the individual character period. We detect this and divide
    by 3 when needed.
    """
    row = img[y_start:y_start + row_height, x_start:x_end]
    col_means = np.mean(row.astype(float), axis=0)
    centered = col_means - np.mean(col_means)

    fft = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered))

    # Search for periods 5-50 px
    min_freq_idx = max(1, int(len(freqs) * (1.0 / 50)))
    max_freq_idx = int(len(freqs) * (1.0 / 5))

    if max_freq_idx <= min_freq_idx:
        return 10.0

    search_range = fft[min_freq_idx:max_freq_idx]
    peak_idx = np.argmax(search_range) + min_freq_idx
    peak_freq = freqs[peak_idx]

    if peak_freq > 0:
        raw_width = 1.0 / peak_freq
        print(f"  FFT raw period: {raw_width:.2f} pixels (freq={peak_freq:.4f})")

        # The hex layout has a 3-char repeating unit (e.g., "4E " = 2 digits + space).
        # If the detected period is in the 25-50px range, it's likely the 3-char byte
        # stride. Divide by 3 to get individual character width.
        # Expected individual char width: 8-18px for typical video frames / screenshots.
        if raw_width > 24:
            char_width = raw_width / 3.0
            print(f"  Detected byte-pair period; char width = {raw_width:.1f}/3 "
                  f"= {char_width:.2f} pixels")
        else:
            char_width = raw_width
            print(f"  Char width: {char_width:.2f} pixels")

        return char_width

    return 10.0  # fallback


# --- Byte grid detection via autocorrelation + brute-force ---

def find_byte_grid(img_gray, y_start, y_end, row_height):
    """
    Find the precise byte grid by autocorrelation and brute-force pattern matching.

    Uses fixed search ranges based on frame resolution, not dependent on any
    prior FFT estimate (which can be unreliable when x-bounds include non-data
    pixels like desktop or dialog borders).

    Returns:
        byte_x_offset (float): x-position of byte 0's first digit
        byte_stride (float): spacing between consecutive bytes
        char_width (float): refined character width (= byte_stride / 3)
        byte_group_gap (float): extra gap at the dash (0 if uniform spacing)
        dash_x (float): derived dash center x-position
    """
    h, w = img_gray.shape

    # Average intensity across multiple data rows for robustness
    n_rows = min(8, max(1, (y_end - y_start) // row_height - 1))
    profile = np.zeros(w, dtype=float)
    count = 0
    for i in range(1, 1 + n_rows):  # skip first row which may be partial
        ry = y_start + i * row_height
        if ry + row_height > h:
            break
        row = img_gray[ry:ry + row_height, :]
        profile += np.mean(row.astype(float), axis=0)
        count += 1
    if count > 0:
        profile /= count

    # Step 1: Find byte_stride via autocorrelation on the hex data area
    # The hex data roughly occupies the middle portion of the frame
    hex_x0 = int(w * 0.2)
    hex_x1 = int(w * 0.7)
    hex_profile = profile[hex_x0:hex_x1]
    centered = hex_profile - np.mean(hex_profile)
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
    if autocorr[0] > 0:
        autocorr /= autocorr[0]  # normalize

    # Fixed search ranges based on 1080p frame resolution:
    # char_width ~10-18px → byte_stride ~30-55px → 4x period ~120-220px
    min_stride = 30
    max_stride = 55
    min_4x = 4 * min_stride   # 120
    max_4x = min(4 * max_stride, len(autocorr) - 1)  # 220

    # Try both 1x and 4x periods, cross-check for consistency
    byte_stride = None
    if max_4x > min_4x:
        search_4x = autocorr[min_4x:max_4x + 1]
        peak_4x = np.argmax(search_4x) + min_4x
        byte_stride_4x = peak_4x / 4.0
        print(f"  Autocorrelation 4x peak at lag={peak_4x} → stride={byte_stride_4x:.2f}px "
              f"(corr={autocorr[peak_4x]:.3f})")

    search_1x = autocorr[min_stride:max_stride + 1]
    peak_1x = np.argmax(search_1x) + min_stride
    byte_stride_1x = float(peak_1x)
    print(f"  Autocorrelation 1x peak at lag={peak_1x} → stride={byte_stride_1x:.2f}px "
          f"(corr={autocorr[peak_1x]:.3f})")

    # Use 4x if available and consistent with 1x (within 2px)
    if max_4x > min_4x and abs(byte_stride_4x - byte_stride_1x) < 2:
        byte_stride = byte_stride_4x  # higher precision
        print(f"  1x and 4x agree → using 4x: {byte_stride:.2f}px")
    else:
        byte_stride = byte_stride_1x
        if max_4x > min_4x:
            print(f"  1x and 4x disagree ({byte_stride_1x:.1f} vs {byte_stride_4x:.1f}) "
                  f"→ using 1x: {byte_stride:.2f}px")
        else:
            print(f"  Using 1x: {byte_stride:.2f}px")

    char_width = byte_stride / 3.0
    print(f"  Autocorrelation byte stride: {byte_stride:.2f}px")
    print(f"  Refined char width: {char_width:.2f}px")

    # Step 2: Find byte_x_offset by brute-force scan
    # For each candidate x0, score = sum of relative darkness at predicted
    # byte positions. Using relative darkness (vs local background) prevents
    # dark non-dialog pixels (desktop, borders) from inflating scores.
    digit_window = max(1, int(round(2 * char_width)))
    bg_margin = max(1, int(round(char_width)))  # tight neighborhood for background

    def score_alignment(x0, gap=0.0):
        """Score a candidate byte_x_offset with optional byte_group_gap."""
        total = 0
        for byte_idx in range(16):
            if byte_idx < 8:
                x = int(round(x0 + byte_idx * byte_stride))
            else:
                x = int(round(x0 + byte_idx * byte_stride + gap))
            xe = x + digit_window
            if x < 0 or xe >= w:
                return -1  # out of bounds
            # Local background: brightest value in neighborhood
            bg_x0 = max(0, x - bg_margin)
            bg_x1 = min(w, xe + bg_margin)
            local_bg = np.max(profile[bg_x0:bg_x1])
            if local_bg < 230:
                return -1  # not on bright dialog background (require white)
            # Relative darkness: how much darker than local background
            mean_val = np.mean(profile[x:xe])
            total += (local_bg - mean_val)
        return total

    # Detect dialog bounds using smoothed brightness profile
    # The dialog has a bright white background; the surrounding desktop is darker.
    smooth_kernel = np.ones(100) / 100
    smooth_profile = np.convolve(profile, smooth_kernel, mode='same')
    dialog_cols = np.where(smooth_profile > 200)[0]
    if len(dialog_cols) > 10:
        dialog_left = int(dialog_cols[0])
        dialog_right = int(dialog_cols[-1])
    else:
        dialog_left = 0
        dialog_right = w
    print(f"  Dialog bounds (smoothed): [{dialog_left}, {dialog_right}]")

    # Constrain search range using the data layout geometry:
    # The hex data (16 bytes) must fit within the dialog, with room for
    # the address column on the left and ASCII column on the right.
    hex_span = int(15 * byte_stride + 2 * char_width)  # total width of 16 bytes
    ascii_width = int(16 * char_width)  # ASCII column = 16 characters
    address_width = int(12 * char_width)  # 10 address chars + separator

    search_x0 = dialog_left + address_width
    search_x1 = max(search_x0 + 1,
                     dialog_right - ascii_width - hex_span + int(byte_stride))
    print(f"  Search range for byte_x_offset: [{search_x0}, {search_x1}]")

    # Test gap=0 (uniform spacing across all 16 bytes)
    best_x0_g0 = search_x0
    best_score_g0 = -1
    for x0 in range(search_x0, search_x1):
        s = score_alignment(x0, gap=0.0)
        if s > best_score_g0:
            best_score_g0 = s
            best_x0_g0 = x0

    # Test gap=char_width (extra space at the dash between groups)
    best_x0_gcw = search_x0
    best_score_gcw = -1
    for x0 in range(search_x0, search_x1):
        s = score_alignment(x0, gap=char_width)
        if s > best_score_gcw:
            best_score_gcw = s
            best_x0_gcw = x0

    print(f"  Gap=0 fit:      score={best_score_g0:.1f} (x0={best_x0_g0})")
    print(f"  Gap=cw fit:     score={best_score_gcw:.1f} (x0={best_x0_gcw})")

    # Choose the better model (gap=cw needs to be significantly better)
    if best_score_gcw > best_score_g0 * 1.05:
        byte_x_offset = float(best_x0_gcw)
        byte_group_gap = char_width
        print(f"  → Using gap=char_width model")
    else:
        byte_x_offset = float(best_x0_g0)
        byte_group_gap = 0.0
        print(f"  → Using gap=0 model (uniform spacing)")

    # Derive dash position from byte grid
    if byte_group_gap > 0:
        # Dash center is between byte 7's trailing space and byte 8's first digit
        byte7_end = byte_x_offset + 7 * byte_stride + 2 * char_width
        byte8_start = byte_x_offset + 8 * byte_stride + byte_group_gap
        dash_x = (byte7_end + byte8_start) / 2.0
    else:
        # Dash replaces the space at character position 23 (center at 23.5)
        dash_x = byte_x_offset + 23.5 * char_width

    print(f"  Byte x offset: {byte_x_offset:.1f}")
    print(f"  Dash x (derived): {dash_x:.1f}")

    return byte_x_offset, byte_stride, char_width, byte_group_gap, dash_x


# --- Compute all byte x-positions ---

def compute_all_byte_positions(byte_x_offset, byte_stride, dash_x, byte_group_gap,
                               char_width):
    """
    Compute the pixel x-position for all 32 hex digits (16 bytes × 2 digits),
    accounting for the dash gap between byte groups 0-7 and 8-15.

    Returns:
        positions: list of 32 (digit_index, x_position) tuples
        byte_centers: list of 16 (byte_index, x_center) tuples
    """
    digit_spacing = char_width  # spacing between the two digits of one byte

    byte_centers = []
    for byte_idx in range(16):
        if byte_idx < 8:
            # First group: bytes 0-7
            x = byte_x_offset + byte_idx * byte_stride
        else:
            # Second group: bytes 8-15 (shifted by dash gap)
            x = byte_x_offset + 8 * byte_stride + byte_group_gap + \
                (byte_idx - 8) * byte_stride
        byte_centers.append((byte_idx, x))

    # Each byte has two hex digits
    positions = []
    for byte_idx, x_center in byte_centers:
        # First digit (high nibble)
        x_d0 = x_center
        # Second digit (low nibble)
        x_d1 = x_center + digit_spacing
        positions.append((byte_idx * 2, x_d0))
        positions.append((byte_idx * 2 + 1, x_d1))

    return positions, byte_centers


# --- Debug visualization ---

def save_debug_images(img_gray, img_color, hex_region, row_height, char_width,
                      grid_origin, row_starts, byte_centers, dash_x):
    """Save debug image showing the detected grid overlay."""
    x_start, y_start, x_end, y_end = hex_region

    # Use color image if available, otherwise convert grayscale
    if img_color is not None:
        vis = img_color.copy()
    else:
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Draw hex region boundary (green)
    cv2.rectangle(vis, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

    # Draw row lines (red)
    gx, gy = grid_origin
    if row_starts is not None:
        for rs in row_starts:
            y = y_start + rs
            cv2.line(vis, (x_start, y), (x_end, y), (0, 0, 255), 1)

    # Draw byte column positions (blue) for first 5 rows
    if byte_centers:
        for byte_idx, bx in byte_centers:
            x = int(round(bx))
            y_bot = min(y_start + row_height * 5, y_end)
            cv2.line(vis, (x, y_start), (x, y_bot), (255, 0, 0), 1)

    # Draw dash position (cyan)
    if dash_x > 0:
        dx = int(round(dash_x))
        cv2.line(vis, (dx, y_start), (dx, min(y_start + row_height * 5, y_end)),
                 (255, 255, 0), 1)

    # Draw character column lines (light blue) for first few rows
    cw = int(round(char_width))
    for col in range(80):
        x = gx + col * cw
        if x > x_end:
            break
        cv2.line(vis, (x, y_start), (x, min(y_start + row_height * 2, y_end)),
                 (200, 100, 0), 1)

    cv2.imwrite('calibration_debug.png', vis)
    print(f"  Debug image saved: calibration_debug.png")


# --- Main calibration ---

def main():
    frame_path = DEFAULT_CALIBRATION_FRAME
    if len(sys.argv) > 1:
        frame_path = sys.argv[1]

    if not os.path.exists(frame_path):
        print(f"Error: frame not found: {frame_path}")
        print(f"Usage: python3 calibrate_grid.py [frame_path]")
        print(f"  Default: {DEFAULT_CALIBRATION_FRAME}")
        print(f"  Alt:     reference/reference_screenshot.png")
        sys.exit(1)

    print(f"Loading frame: {frame_path}")
    img_color = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: cannot read {frame_path}")
        sys.exit(1)

    h, w = img_gray.shape
    print(f"Frame size: {w}x{h}")

    # Step 1: Find hex dump region
    print("\n--- Finding hex dump region ---")
    hex_region = find_hex_dump_region(img_gray, img_color)
    x_start, y_start, x_end, y_end = hex_region
    print(f"  Hex region: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}]")
    print(f"  Region size: {x_end - x_start}x{y_end - y_start}")

    # Step 2: Measure row height
    print("\n--- Measuring row height ---")
    result = measure_row_height(img_gray, y_start, y_end)
    if isinstance(result, tuple) and len(result) == 2:
        row_height, row_starts = result
    else:
        row_height = result
        row_starts = None
    print(f"  Row height: {row_height} pixels")

    # Step 3: Find precise byte grid via autocorrelation + brute-force
    # Uses fixed search ranges (not dependent on FFT estimate which can be
    # unreliable when hex_region x-bounds include non-data pixels).
    print("\n--- Finding byte grid (autocorrelation + pattern matching) ---")
    byte_x_offset, byte_stride, char_width, byte_group_gap, dash_x = \
        find_byte_grid(img_gray, y_start, y_end, row_height)

    # Compute byte area bounds from the grid
    byte_area_x_start = int(round(byte_x_offset))
    if byte_group_gap > 0:
        byte_area_x_end = int(round(byte_x_offset + 8 * byte_stride +
                                    byte_group_gap + 7 * byte_stride +
                                    2 * char_width))
    else:
        byte_area_x_end = int(round(byte_x_offset + 15 * byte_stride +
                                    2 * char_width))
    print(f"  Byte area: x=[{byte_area_x_start}, {byte_area_x_end}]")

    # Step 5: Find grid origin (address column start)
    # Search for first dark text pixels to the left of the byte area
    print("\n--- Finding grid origin ---")
    addr_search_x0 = max(0, byte_area_x_start - int(15 * char_width))
    addr_search_x1 = byte_area_x_start
    if addr_search_x1 > addr_search_x0:
        first_row_strip = img_gray[y_start:y_start + row_height,
                                   addr_search_x0:addr_search_x1]
        col_means = np.mean(first_row_strip.astype(float), axis=0)
        bg_level = np.max(col_means)
        dark_threshold = bg_level * 0.85
        dark_cols = np.where(col_means < dark_threshold)[0]
        if len(dark_cols) > 0:
            grid_x = addr_search_x0 + int(dark_cols[0])
        else:
            grid_x = max(0, byte_area_x_start - int(12 * char_width))
    else:
        grid_x = max(0, byte_area_x_start - int(12 * char_width))
    grid_origin = (grid_x, y_start)
    print(f"  Grid origin: ({grid_x}, {y_start})")

    # Step 6: Compute all byte positions
    print("\n--- Computing byte positions ---")
    digit_positions, byte_centers = compute_all_byte_positions(
        byte_x_offset, byte_stride, dash_x, byte_group_gap, char_width
    )

    # Step 7: Count visible data rows
    # Check each row for hex data presence (high variance in byte area)
    # to avoid counting status bar text below the data area
    print("\n--- Counting visible rows ---")
    max_possible_rows = (y_end - y_start) // row_height
    num_rows = 0
    for r in range(max_possible_rows):
        ry = y_start + r * row_height
        if ry + row_height > img_gray.shape[0]:
            break
        ba_start = max(0, byte_area_x_start)
        ba_end = min(img_gray.shape[1], byte_area_x_end)
        if ba_end <= ba_start:
            break
        row_strip = img_gray[ry:ry + row_height, ba_start:ba_end]
        row_var = np.var(row_strip.astype(float))
        if row_var < 100:  # low variance = no hex data (blank or status area)
            break
        num_rows += 1
    print(f"  Visible data rows: {num_rows}")
    first_row_center_y = y_start + row_height / 2.0

    # Step 10: Build calibration JSON
    params = {
        "frame_width": w,
        "frame_height": h,
        "hex_region": {
            "x_min": int(x_start),
            "x_max": int(x_end),
            "y_min": int(y_start),
            "y_max": int(y_end)
        },
        "row_height": round(float(row_height), 2),
        "first_row_center_y": round(first_row_center_y, 2),
        "visible_rows": int(num_rows),
        "address_x_start": int(grid_origin[0]),
        "address_digits": 10,
        "address_char_spacing": round(float(char_width), 2),
        "byte_x_offset": int(round(byte_x_offset)),
        "byte_stride": round(float(byte_stride), 2),
        "byte_digit_spacing": round(float(char_width), 2),
        "byte_group_gap": round(float(byte_group_gap), 2),
        "dash_x_position": round(float(dash_x), 2),
        "bytes_per_line": 16,
        "cell_width": int(round(char_width)),
        "cell_height": int(row_height),
        "cell_y_offset": 0,
        "cell_x_offset": 0,
        "calibration_frame": os.path.basename(frame_path),
        "notes": "Calibrated from Xeltek SuperPro 6100N Edit Buffer dialog",
        "layout": {
            "address_cols": [0, 10],
            "hex_start_col": 12,
            "hex_byte_stride": 3,
            "hex_bytes_per_line": 16,
            "byte_group_size": 8,
            "dash_between_groups": True
        },
        "byte_positions": [
            {"byte": bc[0], "x": round(bc[1], 1)} for bc in byte_centers
        ]
    }

    if row_starts is not None:
        params["row_y_offsets"] = [int(rs) for rs in row_starts[:num_rows]]

    # Print summary
    print(f"\n--- Calibration Summary ---")
    print(f"  Frame: {frame_path} ({w}x{h})")
    print(f"  Visible rows: {num_rows}")
    print(f"  Row height: {row_height}px")
    print(f"  Char width: {char_width:.2f}px")
    print(f"  Byte stride: {byte_stride:.2f}px")
    print(f"  Dash gap: {byte_group_gap:.1f}px")
    print(f"  Dash x: {dash_x:.0f}")
    print(f"  Grid origin: ({grid_origin[0]}, {grid_origin[1]})")

    # Save calibration JSON
    output_path = 'grid_calibration.json'
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\n  Calibration saved to {output_path}")

    # Save debug visualization
    print("\n--- Saving debug image ---")
    save_debug_images(img_gray, img_color, hex_region, row_height, char_width,
                      grid_origin, row_starts, byte_centers, dash_x)

    return params


if __name__ == '__main__':
    main()
