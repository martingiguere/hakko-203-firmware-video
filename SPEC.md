# Hakko FM-203 Firmware Extraction from Video — Specification

## 1. Overview

Extract and reconstruct the firmware of a **Hakko FM-203** soldering station from a YouTube video recording of the Xeltek SuperPro 6100N programmer software displaying the hex dump of the MCU's flash memory. This follows the same general approach as the [hakko-202-firmware-video](../hakko-202-firmware-video/) project but adapted for different hardware, a different programmer tool UI, and a different video source.

**Goal**: Recover the firmware for the Renesas R5F21258SNFP microcontroller by applying OCR/classification to video frames showing the Xeltek SuperPro hex buffer scrolling through memory, then provide a human-assisted review tool for verification and correction.

### Buffer Size (Confirmed from Reference Screenshot)

The Xeltek SuperPro "Edit Buffer" dialog shows:
- **Buffer range**: `0000000000H – 0000013FFFH`
- **Total buffer size**: `$14000` = **81,920 bytes (80 KB)**
- **Checksum**: `00D2F2FFH` (32-bit)

This 80 KB buffer covers the full `$00000`–`$13FFF` address range, which includes SFR, RAM, reserved areas, AND the program ROM. The 64 KB of program ROM resides at `$04000`–`$13FFF` within this buffer. The first 16 KB (`$00000`–`$03FFF`) contains SFR, RAM, and reserved space — not flash — and its contents represent the chip state at read time, not persistent firmware.

**Implication**: The extraction pipeline must process all 80 KB (5,120 lines of 16 bytes) from the buffer, but the final firmware binary should focus on the 64 KB ROM region (`$04000`–`$13FFF` = 4,096 lines). The non-ROM region (`$00000`–`$03FFF`) may still be useful for understanding the chip's runtime state.

---

## 2. Target Hardware

| Parameter | Value |
|---|---|
| **MCU** | Renesas R5F21258SNFP (R8C/24 Group, R8C/Tiny Series) |
| **Package** | 52-pin TQFP (TQFP-52) |
| **Architecture** | 16-bit CISC, 89 instructions, variable-length encoding (1-5 bytes) |
| **Program ROM** | 64 KB total: 48 KB at `$04000`–`$0FFFF` + 16 KB at `$10000`–`$13FFF` |
| **Internal RAM** | 3 KB (`$00400`–`$00FFF`) |
| **SFR** | 768 bytes (`$00000`–`$002FF`) |
| **Address Space** | 1 MB linear (`$00000`–`$FFFFF`) |
| **Erased Flash State** | `0xFF` |
| **Firmware Version** | V2.00 (per blog post — both units tested) |
| **Source Device** | Hakko FM-203 Soldering Station (main control board B3403) |
| **Programmer** | Xeltek SuperPro 6100N with DX3063 adapter |
| **ISP Protection** | ID code protected (ISP via J4 header blocked; off-board parallel read required) |

### ISP ID Code (Recovered from firmware dump)

Per the blog post, the ID code bytes reside at fixed addresses in the flash:

| Address | Byte |
|---------|------|
| `$FFDF` | `2F` |
| `$FFE3` | `4B` |
| `$FFEB` | `30` |
| `$FFEF` | `30` |
| `$FFF3` | `32` |
| `$FFF7` | `35` |
| `$FFFB` | `36` |

These can be used for validation: once extracted, the firmware at these addresses must contain these exact byte values.

### Memory Map (from R5F21258SNFP datasheet & emulator SPECIFICATION.md)

```
$00000 - $002FF   SFR (Special Function Registers)     768 B
$00300 - $003FF   Reserved                              256 B
$00400 - $00FFF   Internal RAM                          3 KB
$01000 - $03FFF   Reserved                              12 KB
$04000 - $0FFFF   Program ROM (lower 48 KB)             48 KB
$10000 - $13FFF   Program ROM (upper 16 KB)             16 KB
$0FFDC - $0FFFF   Fixed interrupt vector table           36 B
$0FFFC - $0FFFF   Reset vector (20-bit entry point)
```

**Confirmed from reference screenshot**: The Xeltek SuperPro reads the full `$00000`–`$13FFF` range (80 KB) with physical addresses preserved. The buffer addresses correspond directly to MCU physical addresses — no offset translation is needed. The 64 KB of program ROM occupies `$04000`–`$0FFFF` (lower 48 KB) and `$10000`–`$13FFF` (upper 16 KB) within the 80 KB buffer. The reference screenshot shows rows at addresses `$0FF70`–`$10060`, confirming that the address space is contiguous across the `$0FFFF`/`$10000` boundary in the buffer.

### Fixed Interrupt Vector Table (`$0FFDC`–`$0FFFF`)

| Address | Vector |
|---------|--------|
| `$0FFDC` | Undefined instruction |
| `$0FFE0` | Overflow (INTO) |
| `$0FFE4` | BRK instruction |
| `$0FFE8` | Address match |
| `$0FFEC` | Single step |
| `$0FFF0` | Watchdog timer / oscillation stop / voltage monitor |
| `$0FFF4` | Reserved |
| `$0FFF8` | Reserved |
| `$0FFFC` | **Reset vector** (20-bit entry point) |

Each entry is 4 bytes, little-endian, 20-bit address zero-extended.

---

## 3. Video Source

| Property | Value |
|---|---|
| **YouTube URL** | `https://www.youtube.com/watch?v=F3vQnaCocdQ` |
| **Relevant Timestamps** | 13:41 – 24:50 (~11 minutes, 9 seconds) |
| **Content** | Xeltek SuperPro 6100N software showing hex buffer of R5F21258SNFP firmware |
| **Reference Screenshot** | `https://www.stevenrhine.com/wp-content/uploads/2021/08/Hakko-FM-203-Firmware-Backup-TQFP52-Renasas-R5F21258SNFP-Successful-Data.png` |
| **Blog Post** | `https://www.stevenrhine.com/?p=61168` |

### Video Download

```bash
# Download the relevant segment from YouTube using yt-dlp
# Timestamps: 13:41 (821s) to 24:50 (1490s)
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    --download-sections "*821-1490" \
    -o "xeltek_dump.mp4" \
    "https://www.youtube.com/watch?v=F3vQnaCocdQ"
```

Alternatively, download the full video and extract the segment with ffmpeg:

```bash
yt-dlp -f "bestvideo+bestaudio/best" -o "full_video.%(ext)s" \
    "https://www.youtube.com/watch?v=F3vQnaCocdQ"
ffmpeg -i full_video.* -ss 00:13:41 -to 00:24:50 -c copy xeltek_dump.mkv
```

### Frame Extraction

```bash
# Extract frames at native fps (likely 30fps)
mkdir -p frames/
ffmpeg -i xeltek_dump.mkv -vf "fps=30" frames/frame_%05d.png
```

Expected output: ~20,070 frames for the 11m9s segment at 30fps. Actual count will vary based on video encoding.

### Xeltek SuperPro Software UI

The Xeltek SuperPro 6100N uses a Windows-based application with an "Edit Buffer" dialog that displays a hex editor buffer view.

#### Confirmed UI Layout (from reference screenshot)

The reference screenshot (`1063 × 790 px`) reveals the exact layout:

- **Window title**: "Edit Buffer"
- **Column headers**: Green bars labeled `ADDRESS`, `HEX`, `ASCII`
- **Address format**: **10 hex digits** (e.g., `000000FF70`), no separator
- **Row format**:
  ```
  000000FF70   4E FC 00 00 4E FC 00 00-4E FC 00 00 4E FC 00 00   N...N...N...N...
  ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
  ADDRESS(10)  8 HEX BYTES             DASH  8 HEX BYTES          ASCII COLUMN
  ```
- **Byte grouping**: 8 + dash + 8 (the two groups of 8 bytes are separated by a **dash character**, not just whitespace)
- **Visible rows**: ~16 data rows visible in the Edit Buffer dialog
- **Scrollbar**: Vertical scrollbar on the right side of the hex area
- **Bottom status area**:
  - `Address: 000000FF7FH` (cursor position)
  - `Checksum: 00D2F2FFH` (32-bit buffer checksum)
  - `Buffer range: 0000000000H - 0000013FFFH`
  - Checkbox: "Buffer clear on data load" (checked)
  - Checkbox: "Buffer save when exit" (unchecked)
- **Bottom buttons**: Locate, Copy, Fill, Search, Search Next, Radix, Swap, Duplicate, OK

**Important**: The exact pixel positions, character cell dimensions, and spacing must be calibrated from the actual video frames (which may have different resolution and scaling than the blog screenshot). The FM-202 project's `grid_calibration.json` approach should be replicated with new parameters specific to the Xeltek UI.

#### Key Differences from Segger Flasher (FM-202)

| Feature | Segger Flasher (FM-202) | Xeltek SuperPro (FM-203) |
|---------|------------------------|--------------------------|
| Address digits | 6 (`0040F0`) | 10 (`000000FF70`) |
| Address separator | Colon (`:`) | None (space only) |
| Byte grouping | Uniform spacing | 8-dash-8 (`XX XX XX XX-XX XX XX XX`) |
| Visible rows | 16 | ~16 (TBC from video) |
| Column headers | None | Green bars (ADDRESS, HEX, ASCII) |
| Status info | Separate UI area | Below hex area in same dialog |
| Checksum | CRC-16 displayed | 32-bit checksum (`00D2F2FFH`) |

---

## 4. Architecture — Pipeline & Review Tool

The project follows the same phased architecture as the FM-202 project:

```
YouTube Video (Xeltek SuperPro hex buffer)
    │
    ▼
download_video.py + ffmpeg → ~20,000 PNG frames
    │
    ▼
extract_pipeline.py (kNN classification) → Firmware extraction
    │
    ▼
postprocess_firmware.py (merge, filter, fill) → firmware_merged.txt
    │
    ▼
firmware_review_tool/ (Flask web app) → Human review interface
    │
    ▼
firmware_reviewed.txt + hakko_fm203.bin (exports)
```

### Key Differences from FM-202 Project

| Aspect | FM-202 | FM-203 |
|--------|--------|--------|
| **MCU** | M38039FFFP (MELPS 740, 8-bit) | R5F21258SNFP (R8C/24, 16-bit) |
| **Buffer Size** | 60 KB (`$1000`–`$FFFF`) | 80 KB (`$00000`–`$13FFF`) containing 64 KB ROM |
| **Programmer** | Segger Flasher | Xeltek SuperPro 6100N |
| **Hex dump format** | Segger: uniform 16-byte rows | Xeltek: 8-dash-8 grouped bytes |
| **Address format** | 6-digit hex (Segger) | 10-digit hex (Xeltek) |
| **Video duration** | ~7.5 min scrolling | ~11 min of hex display |
| **Video source** | Camera pointed at monitor | Camera pointed at monitor (TBC) |
| **Expected checksum** | CRC-16 `0x0657` | 32-bit `00D2F2FFH` (from Xeltek UI) |
| **Instruction validation** | MELPS 740 opcode table | R8C/Tiny opcode table (from emulator) |
| **Erased flash range** | `$1000`–`$4070`, `$8A50`–`$FFC0` | TBD from extraction |
| **Reference data** | 69 verified lines from screenshot | Reference screenshot available (see below) |

---

## 5. Phase 1: Video Analysis & Grid Calibration

Before building the extraction pipeline, the video must be analyzed to determine the Xeltek SuperPro UI layout.

### 5.1 Initial Frame Analysis

From the first few extracted frames, determine:

1. **Video resolution**: Native resolution of the downloaded video
2. **Frame rate**: Actual vs. nominal fps
3. **Hex dump region**: Bounding box of the hex editor area within the frame
4. **Character grid**: Row height, column positions for address digits and hex bytes
5. **Font metrics**: Character cell dimensions (width × height in pixels)
6. **Address format**: Number of digits, separator style (colon, space, etc.)
7. **Byte grouping**: Whether bytes are grouped in 8+8 or uniform 16
8. **ASCII column position**: Location and whether it's useful for cross-validation
9. **Visible rows per frame**: How many complete rows of data are visible
10. **Toolbar/status bar regions**: Areas to exclude from OCR processing

### 5.2 Reference Screenshot Analysis

The blog post provides a reference screenshot:
`https://www.stevenrhine.com/wp-content/uploads/2021/08/Hakko-FM-203-Firmware-Backup-TQFP52-Renasas-R5F21258SNFP-Successful-Data.png`

Download this screenshot and use it for:
- **Ground-truth data**: Manually transcribe visible hex data for OCR accuracy validation
- **Grid calibration verification**: Compare calibration parameters against a known-good frame
- **Address mapping**: Determine how the Xeltek software maps buffer offsets to physical addresses

### 5.3 Grid Calibration Output

Store calibration in `grid_calibration.json`:

```json
{
  "frame_width": 0,
  "frame_height": 0,
  "hex_region": {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0},
  "row_height": 0.0,
  "first_row_center_y": 0.0,
  "visible_rows": 0,
  "address_x_start": 0,
  "address_digits": 10,
  "address_char_spacing": 0.0,
  "byte_x_offset": 0,
  "byte_stride": 0.0,
  "byte_digit_spacing": 0.0,
  "byte_group_gap": 0.0,
  "dash_x_position": 0.0,
  "bytes_per_line": 16,
  "cell_width": 0,
  "cell_height": 0,
  "cell_y_offset": 0,
  "cell_x_offset": 0,
  "calibration_frame": "frame_NNNNN",
  "notes": "Calibrated from Xeltek SuperPro 6100N Edit Buffer dialog"
}
```

**`byte_group_gap`**: Extra pixel spacing around the dash between byte 7 and byte 8. The Xeltek UI uses an actual dash character as a separator (confirmed from screenshot), so the byte stride is not uniform across all 16 positions.

**`dash_x_position`**: X pixel position of the dash separator between byte groups. Used to skip the dash during character extraction.

**`address_digits`**: Set to 10 (confirmed from reference screenshot: addresses like `000000FF70`).

---

## 6. Phase 2: Extraction Pipeline

### 6.1 Classifier Training

Reuse the FM-202 project's proven kNN approach with 45-dimensional structural features, but train from scratch on Xeltek SuperPro font glyphs:

1. **Find calibration frames**: Identify frames with known data (e.g., matching the reference screenshot, or containing recognizable patterns like the ID code bytes)
2. **Extract character cells**: Using the grid calibration, extract individual character cells from hex digit positions
3. **Build training set**:
   - **Pass 1**: Use Tesseract OCR to label characters in high-confidence frames where addresses are clearly readable
   - **Pass 2**: Use the Pass 1 kNN to label additional frames from diverse timestamps, validated against any known reference data
4. **Feature extraction**: Same 45-dimensional feature vector as FM-202 (horizontal/vertical profiles, quadrant densities, center bar, corners, symmetry, sub-grid)
5. **Classifier**: FastKNNClassifier with k=7, weighted voting

The Xeltek font may differ significantly from the Segger Flasher font — character cell dimensions, stroke weights, and distinguishing features may require tuning the feature extraction parameters.

### 6.2 Frame Processing

For each unique frame:

1. **Deduplication**: Compare hex region pixels to previous frame; skip if difference < threshold
2. **UI state detection**: Detect if the hex buffer area is fully visible (no dialog boxes, menus, or transitions obscuring it)
3. **Address reading**: Extract and decode the address/offset column for each visible row
4. **Address validation**: Verify addresses are sequential (incrementing by `$10` per row)
5. **Byte reading**: Extract and classify all 32 hex digits (16 bytes × 2 digits) per row
6. **Record observations**: Store `(offset, bytes, confidence, frame_id)` tuples

### 6.3 Address Mapping (Confirmed)

The reference screenshot confirms **Option B**: the Xeltek buffer addresses correspond directly to MCU physical addresses. The buffer range is `$00000`–`$13FFF` (80 KB), and addresses shown in the hex view (e.g., `000000FF70`) are the actual physical MCU addresses with leading zeros padded to 10 digits.

- **No offset translation needed**: Buffer address = physical MCU address
- **ROM region within buffer**: `$04000`–`$0FFFF` (lower 48 KB) + `$10000`–`$13FFF` (upper 16 KB)
- **Non-ROM region**: `$00000`–`$03FFF` (SFR, RAM, reserved — 16 KB)

The reference screenshot shows the address `000000FFE0` containing the interrupt vector data, and addresses crossing from `$0FFF0` to `$10000` seamlessly, confirming contiguous physical addressing.

### 6.4 Multi-Frame Voting

Same approach as FM-202:
- Per-byte weighted majority vote across all observations of each address
- Weight = byte_confidence × address_confidence
- Select most-voted value per byte position

### 6.5 Post-Processing

1. **Filter invalid lines**: Non-aligned addresses, ASCII artifacts, implausible byte patterns
2. **Systematic error corrections**: Identify and fix classifier-specific confusion patterns (analogous to FM-202's CF→FF correction)
3. **Overlay reference data**: Verified data always wins
4. **FF-fill erased flash gaps**: Identify regions where both boundary lines are all-FF and fill the gap
5. **ID code validation**: Verify known bytes at ISP ID code addresses
6. **Interrupt vector validation**: Check that the vector table at `$0FFDC`–`$0FFFF` contains plausible 20-bit code addresses

### 6.6 Instruction Validation (R8C/Tiny)

The R5F21258SNFP emulator project includes instruction encoding documentation (`R5F21258SNFP_emulator/INSTRUCTION_ENCODING.md`). This can be used analogously to how the FM-202 project used MELPS 740 opcode tables:

- Validate extracted byte sequences against the R8C/Tiny instruction set (89 instructions)
- Identify impossible opcode sequences that indicate OCR errors
- Suggest corrections based on instruction encoding constraints
- Use recursive descent from the reset vector to trace valid code paths

### 6.7 Output Files

| File | Purpose |
|------|---------|
| `extracted_firmware.txt` | Raw extraction output (before merge) |
| `firmware_merged.txt` | Merged/corrected hex dump |
| `firmware_merged_sources.json` | Per-address source map |
| `hakko_fm203.bin` | 80 KB full buffer binary (or 64 KB ROM-only binary) |
| `fast_knn_classifier.npz` | Trained kNN classifier |
| `grid_calibration.json` | Grid geometry parameters |

### 6.8 Binary Generation

Two binary outputs are generated:

**Full buffer binary (80 KB)** — byte-for-byte image of the entire Xeltek buffer:

```python
# 80 KB buffer image ($00000-$13FFF), pre-filled with 0xFF
buffer = bytearray([0xFF] * 0x14000)  # 81,920 bytes

# Buffer addresses = physical MCU addresses (no translation)
for address, bytes_data in extracted_lines.items():
    for i, b in enumerate(bytes_data):
        buffer[address + i] = b

with open('hakko_fm203_full.bin', 'wb') as f:
    f.write(buffer)
```

**ROM-only binary (64 KB)** — just the program ROM, suitable for programming onto a replacement chip:

```python
# 64 KB ROM image ($04000-$13FFF), pre-filled with 0xFF
rom = bytearray([0xFF] * 0x10000)  # 65,536 bytes

for address, bytes_data in extracted_lines.items():
    if 0x04000 <= address <= 0x13FF0:
        rom_offset = address - 0x04000
        for i, b in enumerate(bytes_data):
            rom[rom_offset + i] = b

with open('hakko_fm203.bin', 'wb') as f:
    f.write(rom)
```

---

## 7. Phase 3: Firmware Review Tool

A Flask/HTML web application for human-assisted review and correction of the extracted firmware, adapted from the FM-202 review tool.

### 7.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (HTML/CSS/JS)                                      │
│  ┌──────────┐ ┌──────────────────────────────────────────┐  │
│  │ Minimap  │ │  Main Review Panel                       │  │
│  │ (pixel   │ │  ┌─────────────────────────────────┐     │  │
│  │  strip)  │ │  │ Offset: 0x00A340  Status: MISS  │     │  │
│  │          │ │  ├─────────────────────────────────┤     │  │
│  │ 5,120 px │ │  │ [A3][3C][01][26][..][..][..]...│     │  │
│  │ tall     │ │  │  16 editable byte cells         │     │  │
│  │          │ │  ├─────────────────────────────────┤     │  │
│  │ click to │ │  │ Frame crops (vertical stack)    │     │  │
│  │ jump     │ │  │  ┌─────────────────────────┐    │     │  │
│  │          │ │  │  │ Frame 0743 [full row]   │    │     │  │
│  │          │ │  │  │ Frame 0744 [full row]   │    │     │  │
│  │          │ │  │  └─────────────────────────┘    │     │  │
│  └──────────┘ └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Stats Bar: Reviewed N/M missing | CRC: 0xXXXX        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │ HTTP (JSON + image serving)
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Flask Backend (Python)                                      │
│  - Serves frame crop images from precomputed cache           │
│  - Provides line data, confidence, OCR readings via JSON API │
│  - Handles save/export                                       │
│  - Computes live CRC                                         │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Data Layer (filesystem)                                     │
│  - crops/<offset>/<frame>.png    (precomputed crop cache)    │
│  - crops/gap/<offset>/<frame>.png (gap context crops)        │
│  - crops/crop_index.json          (frame mapping + readings) │
│  - crops/gap_context_index.json   (gap frame discovery)      │
│  - review_state.json              (working save file)        │
│  - firmware_reviewed.txt          (exported hex dump)        │
│  - firmware_merged.txt            (original baseline, ro)    │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Scope

The tool covers all 5,120 lines in the 80 KB buffer (`$00000`–`$13FFF`, 80 KB / 16 bytes per line). The 4,096 ROM lines (`$04000`–`$13FFF`) are the primary review focus; the 1,024 non-ROM lines (`$00000`–`$03FFF`) are available but lower priority. Lines are presented in prioritized order:

1. **Missing lines** — no extracted data
2. **Low-confidence lines** — extracted but kNN vote margin was narrow
3. **High-confidence lines** — extracted with strong consensus

### 7.3 Precomputation Phase

#### 7.3.1 Frame-to-Offset Mapping

For each unique frame:
1. Read the frame image
2. Detect any UI state issues (dialogs, menus, partial visibility)
3. Use grid calibration to extract the offset column for each visible row
4. Decode the hex offset using the kNN classifier
5. Record mapping: `frame_number → [list of offsets visible]`

#### 7.3.2 Row Cropping

For each (offset, frame) pair:
1. Calculate the Y position using grid calibration
2. Crop the full row: offset column + 16 hex bytes
3. Save as `crops/<offset>/<frame_NNNNN>.png`

#### 7.3.3 OCR Disagreement Data

For each (offset, frame) pair, record the kNN reading and per-byte confidence. Store in `crop_index.json` (same format as FM-202 project).

#### 7.3.4 Gap Frame Discovery

For missing offsets where the standard pipeline found no data:
1. Search for neighboring offsets (±8 rows) that have frame data
2. Score candidate frames by proximity + sharpness
3. Generate 9-row context crops with CLAHE enhancement
4. Save to `crops/gap/<offset>/` with metadata in `gap_context_index.json`

### 7.4 UI Layout

Follows the same design as the FM-202 review tool (see `hakko-202-firmware-video/claude/SPEC_firmware_review_tool.md` for the full specification). Key adaptations:

- **Minimap height**: 5,120 pixels (one per buffer line, $00000–$13FFF), with the non-ROM region ($00000–$03FFF) visually distinguished (dimmed or separate color)
- **Address display**: Shows the physical MCU address (e.g., `$0A340`) with a region indicator (`ROM` or `SFR/RAM`)
- **Byte cell alignment**: Uses Xeltek-specific grid constants (byte stride, group gap)
- **ID code validation indicator**: Highlight the 7 known ID code byte positions with a special badge when their values match or mismatch the expected values

### 7.5 Stats Bar

- **Review progress**: `Reviewed: N/M missing | N/M suspect | N/M confident`
- **Current checksum**: Live 32-bit checksum computation to match the Xeltek-reported value `00D2F2FFH` (algorithm TBD — likely a simple byte sum or CRC-32; must be determined by testing against known data)
- **ID code status**: `ID: 7/7 match` or `ID: 5/7 match` — quick indicator of known-byte accuracy
- **Address jump**: Text input for jumping to a specific offset
- **Save/Export buttons**

### 7.6 Keyboard Shortcuts

Same as FM-202 review tool:

| Key | Action |
|-----|--------|
| `Enter` | Accept current line and advance |
| `Escape` | Skip current line and advance |
| `F` | Flag current line and advance |
| `Left` / `[` | Navigate to previous line |
| `Right` / `]` | Navigate to next line |
| `Tab` / `Shift+Tab` | Move between byte cells |
| `Ctrl+S` | Save to disk |
| `Ctrl+G` | Focus address jump input |
| `+` / `-` | Zoom frame crops |
| `R` | Reset line to baseline |
| `1`–`9` | Jump to frame crop #N |

### 7.7 Save/Export System

- **Save**: `review_state.json` — preserves all review state, resumable across sessions
- **Export Hex**: `firmware_reviewed.txt` — hex dump in standard format
- **Export Binary (ROM)**: `hakko_fm203_reviewed.bin` — 64 KB ROM-only binary (`$04000`–`$13FFF`)
- **Export Binary (Full)**: `hakko_fm203_full_reviewed.bin` — 80 KB full buffer binary (`$00000`–`$13FFF`)

### 7.8 Flask Backend API

Same endpoints as FM-202 review tool, with path/naming adapted:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main HTML page |
| `GET` | `/api/line/<offset>` | Line data: bytes, status, confidence, frames |
| `GET` | `/api/frames/<offset>` | Frame list + per-frame kNN readings |
| `GET` | `/api/crop/<offset>/<frame>` | Serve crop PNG |
| `GET` | `/api/gap_frames/<offset>` | Gap context candidates |
| `GET` | `/api/gap_context/<offset>/<frame>` | Gap context crop PNG |
| `GET` | `/api/gaps` | All gaps grouped by gap_id |
| `GET` | `/api/minimap` | Minimap data array |
| `GET` | `/api/stats` | Review statistics + CRC |
| `POST` | `/api/line/<offset>` | Update line data |
| `POST` | `/api/save` | Write review_state.json |
| `POST` | `/api/export/hex` | Generate firmware_reviewed.txt |
| `POST` | `/api/export/binary` | Generate hakko_fm203_reviewed.bin (ROM) + full buffer |
| `GET` | `/api/settings` | UI settings |
| `POST` | `/api/settings` | Update UI settings |

---

## 8. Validation Strategy

### 8.1 Known-Byte Validation

The ISP ID code provides 7 known bytes at specific addresses. These serve as ground truth for accuracy measurement (analogous to FM-202's authoritative reference transcription, but much smaller).

### 8.2 Reference Screenshot

The reference screenshot from the blog post should be downloaded, manually transcribed, and used as additional ground truth. This provides a larger set of verified bytes than the ID code alone.

### 8.3 Interrupt Vector Validation

The fixed interrupt vector table at `$0FFDC`–`$0FFFF` must contain valid 20-bit code addresses pointing into the ROM region (`$04000`–`$13FFF`). Any vector pointing outside this range (except `$00000` for unused vectors) indicates extraction errors.

### 8.4 R8C/Tiny Instruction Validation

Use the instruction encoding data from `R5F21258SNFP_emulator/INSTRUCTION_ENCODING.md` to:
- Validate opcode sequences in extracted code regions
- Identify invalid instruction encodings that suggest OCR errors
- Trace execution paths from the reset vector

### 8.5 Cross-Unit Validation

The blog post mentions both units tested ran Firmware V2.00. If firmware from a second unit becomes available, byte-for-byte comparison would identify extraction errors vs. genuine firmware differences.

### 8.6 Checksum Validation

The Xeltek SuperPro displays a 32-bit checksum in the Edit Buffer dialog: **`00D2F2FFH`**. This serves as the target checksum for the full 80 KB buffer (`$00000`–`$13FFF`).

**Checksum algorithm identification**: The exact algorithm must be determined. Common possibilities:
- Simple 32-bit byte sum (most likely for Xeltek — sum all bytes, truncate to 32 bits)
- CRC-32 (IEEE 802.3)
- CRC-32/JAMCRC

The algorithm can be identified by computing candidate checksums against the reference screenshot's visible data and cross-referencing with the displayed value. Once identified, the checksum provides a strong end-to-end validation of the full extracted buffer.

---

## 9. Project File Structure

```
hakko-203-firmware-video/
├── SPEC.md                          # This specification
├── README.md                        # Quick start guide
├── requirements.txt                 # Python dependencies
├── download_video.py                # YouTube download + segment extraction
│
├── frames/                          # Extracted video frames (gitignored)
│   ├── frame_00001.png
│   └── ...
│
├── reference/                       # Reference material
│   ├── reference_screenshot.png     # Downloaded blog screenshot
│   └── reference_transcription.txt  # Hand-verified hex from screenshot
│
├── grid_calibration.json            # Xeltek UI grid parameters
├── extract_pipeline.py              # Main extraction pipeline
├── template_matcher.py              # kNN classifier (adapted from FM-202)
├── postprocess_firmware.py          # Filtering, correction, merge, binary gen
├── r8c_opcode_table.py              # R8C/Tiny opcode table for validation
├── r8c_disassembler.py              # R8C/Tiny disassembler for validation
│
├── extracted_firmware.txt           # Raw kNN extraction output
├── firmware_merged.txt              # Final merged hex dump
├── firmware_merged_sources.json     # Per-address source tracking
├── hakko_fm203.bin                  # 64 KB ROM-only firmware binary
├── hakko_fm203_full.bin             # 80 KB full buffer binary
├── fast_knn_classifier.npz          # Trained kNN model
│
├── firmware_review_tool/            # Flask review application
│   ├── app.py                       # Flask backend
│   ├── precompute.py                # Standard row crop generation
│   ├── precompute_gaps.py           # Gap context crop generation
│   ├── crc.py                       # CRC computation
│   ├── templates/
│   │   └── index.html               # Single-page frontend
│   └── test_review_tool.py          # Test suite
│
├── crops/                           # Precomputed frame crops (gitignored)
│   ├── crop_index.json
│   ├── gap_context_index.json
│   ├── <offset>/
│   │   └── frame_NNNNN.png
│   └── gap/
│       └── <offset>/
│           └── frame_NNNNN.png
│
├── review_state.json                # Review progress (gitignored)
├── firmware_reviewed.txt            # Exported reviewed hex dump
└── hakko_fm203_reviewed.bin         # Exported reviewed binary
```

---

## 10. Implementation Phases

### Phase 1: Setup & Calibration
1. Set up project directory and Python environment
2. Download video segment and extract frames
3. Download reference screenshot
4. Analyze Xeltek SuperPro UI layout from frames
5. Build grid calibration JSON
6. Manually transcribe reference screenshot for ground truth

### Phase 2: Extraction Pipeline
1. Adapt `template_matcher.py` for Xeltek font characteristics
2. Build kNN classifier from labeled frames (two-pass training)
3. Implement frame processing pipeline (dedup, address read, byte read, voting)
4. Implement post-processing (filtering, corrections, FF-fill, binary generation)
5. Validate against reference data and ID code bytes

### Phase 3: Review Tool
1. Adapt `precompute.py` for Xeltek grid layout
2. Adapt `precompute_gaps.py` for gap context generation
3. Adapt `app.py` Flask backend (new paths, address mapping, CRC)
4. Adapt `index.html` frontend (byte cell alignment, address display, minimap sizing)
5. Add region indicator for ROM vs. non-ROM addresses
6. Add ID code validation indicator

### Phase 4: Review & Export
1. Run human review using the web tool
2. Iterate on corrections using frame crops as visual reference
3. Export final firmware binary
4. Validate against all known constraints (ID code, vectors, instruction validity)
5. Package as release tarball for standalone use

---

## 11. Code Reuse from FM-202 Project

The following components can be reused with adaptation:

| FM-202 Component | Reuse Level | Adaptation Needed |
|---|---|---|
| `template_matcher.py` (FastKNNClassifier) | High — core algorithm reusable | Retrain on Xeltek fonts; may need feature tuning |
| `extract_pipeline.py` (frame processing) | Medium — structure reusable | New grid calibration, different address format, no window-shift detection |
| `postprocess_firmware.py` (merge/filter) | Medium — logic reusable | Different address ranges, ROM size, error patterns |
| `firmware_review_tool/app.py` | High — architecture reusable | New paths, address mapping, CRC, minimap sizing |
| `firmware_review_tool/precompute.py` | Medium — structure reusable | New grid calibration, no window-shift detection (TBD) |
| `firmware_review_tool/precompute_gaps.py` | High — fully reusable | Minor path/size adjustments |
| `firmware_review_tool/crc.py` | Medium | Different CRC algorithm may be needed |
| `firmware_review_tool/templates/index.html` | High | UI tweaks for dual-address display, ID code indicator |
| `firmware_review_tool/test_review_tool.py` | Medium | New test data, address ranges |

Components that are **not** reusable:
- Grid calibration values (completely different UI)
- Trained kNN classifier (different font)
- Window-shift detection (Xeltek may not have this issue)
- MELPS 740 opcode/disassembler (replaced by R8C/Tiny equivalents)

---

## 12. Dependencies

### Python Packages

```
opencv-python>=4.8
numpy>=1.24
scipy>=1.10
Pillow>=9.0
Flask>=2.3
pytesseract>=0.3.10  # For Pass 1 classifier training
yt-dlp>=2024.0       # For video download
```

### System Dependencies

```
tesseract-ocr        # OCR for initial classifier training
ffmpeg / ffprobe      # Frame extraction
```

---

## 13. Related Projects

| Project | Path | Relationship |
|---------|------|--------------|
| Hakko FM-202 Firmware Video | `../hakko-202-firmware-video/` | Template project — same approach, different hardware |
| R5F21258SNFP Emulator | `../R5F21258SNFP_emulator/` | Target MCU emulator — provides instruction encodings & datasheets |
| Hakko FM-202 Flash R/W | `../hakko-202-flash-rw/` | Related hardware tools |
| M38039FFFP Emulator | `../M38039FFFP_emulator/` | FM-202 MCU emulator (sibling project) |

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Xeltek font is harder to classify than Segger | Lower extraction accuracy | Expand training set; tune features for Xeltek font |
| Video is lower quality / more camera shake | Blurrier frames, more missing data | Multi-frame voting compensates; review tool for manual correction |
| Buffer offset ≠ physical address mapping is complex | Wrong byte placement in binary | Validate against known ID code bytes early in pipeline |
| Checksum algorithm unknown | Cannot validate until algorithm identified | Test common algorithms (byte sum, CRC-32) against reference data; the 32-bit value `00D2F2FFH` is visible in the UI |
| Large erased flash regions obscure actual code coverage | False sense of completion | Track FF-fill vs. observed data separately in source map |
| Video has UI dialogs/popups during scrolling | Frames with obscured hex data | Detect and skip frames with non-standard UI state |
| 11+ minutes of video produces 20K+ frames | Slow processing | Frame deduplication; parallel processing where possible |

---

## 15. Success Criteria

1. **Coverage**: ≥95% of firmware address lines have extracted data (before FF-fill)
2. **ID code match**: All 7 known ID code bytes match expected values
3. **Vector table validity**: All interrupt vectors point to valid ROM addresses
4. **Instruction validity**: Disassembly from reset vector produces valid R8C/Tiny instructions with no illegal opcodes in traced code paths
5. **Reference match**: 100% accuracy on manually transcribed reference screenshot bytes
6. **Checksum match**: 32-bit checksum of the full 80 KB buffer matches Xeltek-reported `00D2F2FFH`
7. **Binary output**: Valid 64 KB ROM binary (`hakko_fm203.bin`) that can be programmed onto a replacement R5F21258SNFP, plus full 80 KB buffer binary

---

## 16. Non-Goals (Out of Scope)

- Running the extracted firmware in the R5F21258SNFP emulator (separate project)
- Reverse engineering the firmware logic
- Modifying the firmware
- Multi-user collaboration on the review tool
- Mobile/responsive layout
- Automated re-reading from the physical MCU
