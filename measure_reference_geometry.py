#!/usr/bin/env python3
"""
Precisely measure the geometry of the reference screenshot for cropping purposes.

Analyzes reference/reference_screenshot.png to determine:
  - Row y-centers for all 16 data rows ($0FF70 through $10060)
  - Row spacing
  - Hex data area x-bounds (for cropping off address and ASCII columns)
  - Byte column centers and byte stride

Methodology:
  - Rows: detected by scanning for horizontal bands where average intensity
    drops below 254 (text ink on white background), giving exact text-band
    boundaries. Row centers are the midpoints of these bands.
  - Bytes: detected by scanning horizontally within row 0's text band for
    contiguous ink runs (intensity < 250), then mapping those runs to the
    known hex content "4E FC 00 00 4E FC 00 00-4E FC 00 00 4E FC 00 00".
"""

import cv2
import numpy as np

IMAGE_PATH = 'reference/reference_screenshot.png'

# ── Load image ─────────────────────────────────────────────────────────────

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if img_gray is None:
    raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")

h, w = img_gray.shape
print(f"Image dimensions: {w} x {h}  (width x height)")
print()

# ── Detect green header bars ──────────────────────────────────────────────

hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))

row_green = np.sum(green_mask > 0, axis=1)
green_rows = np.where(row_green > w * 0.05)[0]

if len(green_rows) > 0:
    header_top = int(green_rows[0])
    header_bottom = int(green_rows[-1]) + 1

    # Find separate bars by column-wise green coverage
    header_mask = green_mask[header_top:header_bottom, :]
    col_green = np.sum(header_mask > 0, axis=0)
    min_coverage = max((header_bottom - header_top) * 0.15, 3)
    green_cols = np.where(col_green > min_coverage)[0]

    bars = []
    if len(green_cols) > 0:
        bar_start = green_cols[0]
        for i in range(1, len(green_cols)):
            if green_cols[i] - green_cols[i - 1] > 15:
                bars.append((int(bar_start), int(green_cols[i - 1])))
                bar_start = green_cols[i]
        bars.append((int(bar_start), int(green_cols[-1])))

    bar_names = ['ADDRESS', 'HEX', 'ASCII']
    print(f"Green header bars: rows {header_top} to {header_bottom - 1}")
    for i, bar in enumerate(bars):
        name = bar_names[i] if i < len(bar_names) else f'BAR{i}'
        print(f"  {name} bar: x=[{bar[0]}, {bar[1]}]  width={bar[1] - bar[0] + 1}")
else:
    header_bottom = 0
    bars = []
    print("WARNING: No green header bars detected")

print()

# ── Find row y-centers via text-band detection ────────────────────────────
#
# Each data row is a horizontal band of text (dark pixels on white bg).
# We scan vertically through the hex data columns, looking for contiguous
# runs of rows where average intensity < 254 (text present).

# Use the HEX bar x-range for intensity averaging
if len(bars) >= 2:
    hex_x0, hex_x1 = bars[1]
else:
    hex_x0, hex_x1 = 190, 756

print(f"Scanning for text bands in x=[{hex_x0}, {hex_x1}] ...")
print()

row_y_ranges = []
in_text = False
text_start = 0

for y in range(header_bottom, int(h * 0.70)):
    avg = np.mean(img_gray[y, hex_x0:hex_x1].astype(float))
    if avg < 254 and not in_text:
        in_text = True
        text_start = y
    elif avg >= 254 and in_text:
        in_text = False
        text_end = y - 1
        text_height = text_end - text_start + 1
        # Only count bands tall enough to be text rows (>= 8px)
        if text_height >= 8:
            row_y_ranges.append((text_start, text_end))

if in_text:
    text_end = int(h * 0.70) - 1
    if text_end - text_start + 1 >= 8:
        row_y_ranges.append((text_start, text_end))

addresses = [
    "$0FF70", "$0FF80", "$0FF90", "$0FFA0",
    "$0FFB0", "$0FFC0", "$0FFD0", "$0FFE0",
    "$0FFF0", "$10000", "$10010", "$10020",
    "$10030", "$10040", "$10050", "$10060",
]

print(f"Found {len(row_y_ranges)} text bands:")
print()
print("=== ROW Y-CENTERS ===")

row_centers = []
for i, (ys, ye) in enumerate(row_y_ranges):
    center = (ys + ye) / 2.0
    height = ye - ys + 1
    row_centers.append(center)
    addr = addresses[i] if i < len(addresses) else f"row {i}"
    spacing_str = ''
    if i > 0:
        spacing_str = f'  spacing={center - row_centers[i - 1]:.1f}'
    print(f"  Row {i:2d} ({addr}):  y=[{ys}, {ye}]  height={height}px  "
          f"center={center:.1f}{spacing_str}")

print()

# Only use the first 16 rows for spacing/geometry (row 16+ may be borders/UI)
if len(row_centers) > 16:
    extra = len(row_centers) - 16
    print(f"  Note: {extra} extra band(s) detected beyond 16 data rows (likely UI elements)")
    row_centers = row_centers[:16]
    row_y_ranges = row_y_ranges[:16]

# Row spacing
spacings = [row_centers[i] - row_centers[i - 1] for i in range(1, len(row_centers))]
print("=== ROW SPACING ===")
print(f"  Individual spacings: {', '.join(f'{s:.1f}' for s in spacings)}")
print(f"  Mean:   {np.mean(spacings):.3f}")
print(f"  Median: {np.median(spacings):.3f}")
print(f"  Min:    {np.min(spacings):.3f}")
print(f"  Max:    {np.max(spacings):.3f}")
print(f"  Stddev: {np.std(spacings):.3f}")
print()

REF_FIRST_ROW_Y = row_centers[0]
REF_ROW_SPACING = float(np.mean(spacings))

# ── Find byte positions via ink-run analysis ──────────────────────────────
#
# Row 0 ($0FF70) contains: 4E FC 00 00 4E FC 00 00-4E FC 00 00 4E FC 00 00
# We scan horizontally within the row 0 text band, averaged vertically,
# looking for contiguous ink runs (intensity < 250).

print("=== BYTE POSITION ANALYSIS ===")
print(f"  Analyzing row 0 text band: y=[{row_y_ranges[0][0]}, {row_y_ranges[0][1]}]")
print()

# Average intensity across the text band for each x
row0_strip = np.mean(img_gray[row_y_ranges[0][0]:row_y_ranges[0][1] + 1, :].astype(float),
                     axis=0)

# Find contiguous ink runs in the hex data area (x=170 to 770)
ink_runs = []
in_ink = False
ink_start = 0
for x in range(170, 770):
    if row0_strip[x] < 250 and not in_ink:
        in_ink = True
        ink_start = x
    elif row0_strip[x] >= 250 and in_ink:
        in_ink = False
        ink_runs.append((ink_start, x - 1))
if in_ink:
    ink_runs.append((ink_start, 769))

print(f"  Found {len(ink_runs)} ink runs in row 0:")
for i, (s, e) in enumerate(ink_runs):
    center = (s + e) / 2.0
    gap_str = ''
    if i > 0:
        gap_str = f'  gap_before={s - ink_runs[i - 1][1] - 1}px'
    print(f"    Run {i:2d}: x=[{s:3d}, {e:3d}]  width={e - s + 1:2d}px  "
          f"center={center:6.1f}{gap_str}")

print()

# Map ink runs to bytes.
# Row 0: "4E FC 00 00 4E FC 00 00-4E FC 00 00 4E FC 00 00"
#
# "4E" produces one merged run (~22px) because the characters touch/overlap.
# "FC" produces one merged run (~24px) for the same reason.
# "00" produces two separate runs (~11px each, gap=1px) because '0' glyphs
#      are separate.
# The "-" (dash) produces one run (~11px).
#
# The 4-byte pattern "4E FC 00 00" produces runs:
#   Run A: 4E (1 run, ~22px)
#   Run B: FC (1 run, ~24px)
#   Run C: 0  (first digit of 00)
#   Run D: 0  (second digit of 00, gap=1px from C)
#   Run E: 0  (first digit of 00)
#   Run F: 0  (second digit of 00, gap=1px from E)
# = 6 runs per 4-byte group
#
# Layout:
#   Runs 0-5:   bytes 0-3 (4E FC 00 00)
#   Runs 6-11:  bytes 4-7 (4E FC 00 00)
#   Run 12:     dash (-)
#   Runs 13-18: bytes 8-11 (4E FC 00 00)
#   Runs 19-24: bytes 12-15 (4E FC 00 00)
#
# From the runs, we can extract byte left/right edges:

# Map: (run_indices_for_byte, byte_index)
byte_run_map = [
    # Group A: bytes 0-7
    ([0], 0),       # 4E = 1 merged run
    ([1], 1),       # FC = 1 merged run
    ([2, 3], 2),    # 00 = 2 runs (gap=1px between digits)
    ([4, 5], 3),    # 00 = 2 runs
    ([6], 4),       # 4E
    ([7], 5),       # FC
    ([8, 9], 6),    # 00
    ([10, 11], 7),  # 00
    # Dash at run 12
    # Group B: bytes 8-15
    ([13], 8),      # 4E
    ([14], 9),      # FC
    ([15, 16], 10), # 00
    ([17, 18], 11), # 00
    ([19], 12),     # 4E
    ([20], 13),     # FC
    ([21, 22], 14), # 00
    ([23, 24], 15), # 00
]

byte_left_edges = []
byte_right_edges = []
byte_centers_list = []

print("  Byte positions (from ink-run mapping):")
for run_indices, byte_idx in byte_run_map:
    left = ink_runs[run_indices[0]][0]
    right = ink_runs[run_indices[-1]][1]
    center = (left + right) / 2.0
    byte_left_edges.append(left)
    byte_right_edges.append(right)
    byte_centers_list.append(center)
    run_desc = '+'.join(str(r) for r in run_indices)
    print(f"    Byte {byte_idx:2d}: runs[{run_desc:5s}] -> x=[{left:3d}, {right:3d}]  "
          f"width={right - left + 1:2d}px  center={center:.1f}")

print()

# Dash position
dash_left = ink_runs[12][0]
dash_right = ink_runs[12][1]
dash_center = (dash_left + dash_right) / 2.0
print(f"  Dash: x=[{dash_left}, {dash_right}]  width={dash_right - dash_left + 1}px  "
      f"center={dash_center:.1f}")
print()

# ── Compute byte stride ──────────────────────────────────────────────────

print("=== BYTE STRIDE ANALYSIS ===")

all_strides = [byte_left_edges[i] - byte_left_edges[i - 1]
               for i in range(1, 16)]
print(f"  Left-edge strides: {all_strides}")

# Stride pattern repeats every 4 bytes: 35, 36, 36, 37
stride_pattern = all_strides[:4]
print(f"  4-byte pattern: {stride_pattern} (sum={sum(stride_pattern)})")

# The pattern is identical across all groups:
for g in range(0, 12, 4):
    group = all_strides[g:g + 4] if g + 4 <= len(all_strides) else all_strides[g:]
    print(f"    Strides [{g}:{g + len(group)}]: {group}")

# Overall average stride
total_span = byte_left_edges[15] - byte_left_edges[0]
avg_stride = total_span / 15.0
print(f"  Total span (byte 0 to byte 15 left edges): {total_span}px")
print(f"  Average stride: {avg_stride:.4f}px")

# Note: the stride 7->8 includes the dash character, but it equals 37
# (same as strides 3->4, 11->12), confirming the dash occupies exactly
# one character cell. There is NO extra gap.
print(f"  Stride 7->8 (across dash): {byte_left_edges[8] - byte_left_edges[7]}")
print(f"  Stride 3->4 (same position in pattern): {byte_left_edges[4] - byte_left_edges[3]}")
print(f"  Stride 11->12 (same position in pattern): {byte_left_edges[12] - byte_left_edges[11]}")
print(f"  -> Dash occupies exactly one character cell (no extra gap)")
print()

byte_stride = avg_stride
char_width = byte_stride / 3.0

# Center-to-center stride
center_strides = [byte_centers_list[i] - byte_centers_list[i - 1]
                  for i in range(1, 16)]
avg_center_stride = np.mean(center_strides)
print(f"  Center-to-center strides: "
      f"{', '.join(f'{s:.1f}' for s in center_strides)}")
print(f"  Average center-to-center stride: {avg_center_stride:.4f}px")
print()

# ── Compute crop bounds ──────────────────────────────────────────────────

hex_ink_left = byte_left_edges[0]    # 193
hex_ink_right = byte_right_edges[15]  # 754
CROP_MARGIN = 5

REF_CROP_X1 = hex_ink_left - CROP_MARGIN
REF_CROP_X2 = hex_ink_right + CROP_MARGIN

print("=== CROP BOUNDS ===")
print(f"  Hex data ink area: x=[{hex_ink_left}, {hex_ink_right}]  "
      f"width={hex_ink_right - hex_ink_left + 1}px")
print(f"  Crop margin: {CROP_MARGIN}px on each side")
print(f"  REF_CROP_X1 = {REF_CROP_X1}")
print(f"  REF_CROP_X2 = {REF_CROP_X2}")
print(f"  Crop width = {REF_CROP_X2 - REF_CROP_X1}px")
print()

# ── Verification ─────────────────────────────────────────────────────────

print("=== VERIFICATION ===")

# Verify rows 0-5 are pixel-identical (all contain "4E FC 00 00" x4)
print("  Row-to-row pixel identity (rows 0-5 should be identical):")
ref_strip = np.mean(img_gray[row_y_ranges[0][0]:row_y_ranges[0][1] + 1,
                              hex_x0:hex_x1].astype(float), axis=0)
for i in range(6):
    ys, ye = row_y_ranges[i]
    test_strip = np.mean(img_gray[ys:ye + 1, hex_x0:hex_x1].astype(float), axis=0)
    max_diff = np.max(np.abs(test_strip - ref_strip))
    print(f"    Row {i} vs row 0: max pixel diff = {max_diff:.2f}  "
          f"({'IDENTICAL' if max_diff < 1 else 'DIFFERENT'})")

# Verify bytes 0, 4, 8, 12 are identical (all 0x4E)
print()
print("  Byte identity check (bytes 0, 4, 8, 12 should all be 0x4E):")
for byte_idx in [0, 4, 8, 12]:
    bl = byte_left_edges[byte_idx]
    br = byte_right_edges[byte_idx]
    avg = np.mean(img_gray[row_y_ranges[0][0]:row_y_ranges[0][1] + 1,
                           bl:br + 1].astype(float))
    print(f"    Byte {byte_idx:2d}: x=[{bl}, {br}]  avg_intensity={avg:.1f}")

print()

# ── Final summary ────────────────────────────────────────────────────────

print("=" * 70)
print("FINAL MEASUREMENTS SUMMARY")
print("=" * 70)
print()
print(f"  Image size:          {w} x {h}  (width x height)")
print(f"  Data rows:           {len(row_centers)}")
print(f"  Row text height:     {row_y_ranges[0][1] - row_y_ranges[0][0] + 1} px")
print(f"  Inter-row gap:       "
      f"{row_y_ranges[1][0] - row_y_ranges[0][1] - 1} px")
print()
print(f"  REF_FIRST_ROW_Y   = {REF_FIRST_ROW_Y:.1f}    "
      f"# y-center of first data row ($0FF70)")
print(f"  REF_ROW_SPACING   = {REF_ROW_SPACING:.1f}     "
      f"# pixels between row centers (exact)")
print(f"  REF_CROP_X1       = {REF_CROP_X1}       "
      f"# left edge of hex data crop")
print(f"  REF_CROP_X2       = {REF_CROP_X2}       "
      f"# right edge of hex data crop")
print()
print(f"  byte_x_offset     = {byte_left_edges[0]}       "
      f"# x of byte 0 first digit (left edge)")
print(f"  byte_stride       = {byte_stride:.4f}  "
      f"# mean stride between byte left edges")
print(f"  char_width        = {char_width:.4f}  "
      f"# = byte_stride / 3")
print(f"  byte_group_gap    = 0.0     "
      f"# no extra gap at dash")
print(f"  dash_center_x     = {dash_center:.1f}    "
      f"# center of dash glyph")
print()
print(f"  Row 0 text band:   y=[{row_y_ranges[0][0]}, {row_y_ranges[0][1]}]")
print(f"  Row 15 text band:  y=[{row_y_ranges[15][0]}, {row_y_ranges[15][1]}]")
print(f"  Green header:      y=[{header_top}, {header_bottom - 1}]")
print()

print("=" * 70)
print("COPY-PASTE CONSTANTS")
print("=" * 70)
print()
print(f"REF_FIRST_ROW_Y = {REF_FIRST_ROW_Y:.1f}    "
      f"# y-center of first data row ($0FF70)")
print(f"REF_ROW_SPACING = {REF_ROW_SPACING:.1f}     "
      f"# pixels between row centers (exact)")
print(f"REF_CROP_X1     = {REF_CROP_X1}       "
      f"# left edge of hex data crop (5px margin)")
print(f"REF_CROP_X2     = {REF_CROP_X2}       "
      f"# right edge of hex data crop (5px margin)")
print(f"BYTE_0_CENTER_X = {byte_centers_list[0]:.1f}    "
      f"# center of byte 0 in original image")
print(f"BYTE_0_LEFT_X   = {byte_left_edges[0]}       "
      f"# left edge of byte 0")
print(f"BYTE_STRIDE     = {byte_stride:.4f}  "
      f"# mean stride between byte left edges")
print(f"CHAR_WIDTH      = {char_width:.4f}  "
      f"# = BYTE_STRIDE / 3")
print(f"BYTE_GROUP_GAP  = 0.0     "
      f"# no extra gap at dash (dash = 1 char cell)")
print(f"DASH_CENTER_X   = {dash_center:.1f}    "
      f"# center of dash glyph")
print()
print("# Byte center x-positions (in original image coordinates):")
print("BYTE_CENTERS = [")
for i, center in enumerate(byte_centers_list):
    comma = ',' if i < 15 else ''
    print(f"    {center:6.1f}{comma}  # byte {i:2d}  "
          f"(left={byte_left_edges[i]:3d}, right={byte_right_edges[i]:3d})")
print("]")
print()

# Byte left edges for reference
print("# Byte left edges (for precise digit extraction):")
print(f"BYTE_LEFT_EDGES = {byte_left_edges}")
print()
print("# Byte right edges:")
print(f"BYTE_RIGHT_EDGES = {byte_right_edges}")
print()

# Stride pattern
print(f"# Stride pattern (repeating every 4 bytes): {stride_pattern}")
print(f"# Stride across dash (7->8): "
      f"{byte_left_edges[8] - byte_left_edges[7]} "
      f"(same as 3->4: {byte_left_edges[4] - byte_left_edges[3]})")
print()
print("Done.")
