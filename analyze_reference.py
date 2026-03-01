#!/usr/bin/env python3
"""
Verify the reference transcription against the reference screenshot.

Loads reference/reference_screenshot.png, applies green-header detection and
row measurement (same algorithms as calibrate_grid.py), extracts visible text
metrics, and cross-validates hex values against the ASCII column.

Also validates the ISP ID code bytes and reports match/mismatch.
"""

import cv2
import numpy as np
import os
import sys

# ISP ID code bytes from SPEC.md Section 2
ISP_ID_CODE = {
    0xFFDF: 0x2F,  # blog post value (screenshot may differ)
    0xFFE3: 0x4B,
    0xFFEB: 0x30,
    0xFFEF: 0x30,
    0xFFF3: 0x32,
    0xFFF7: 0x35,
    0xFFFB: 0x36,
}

REFERENCE_IMAGE = 'reference/reference_screenshot.png'
REFERENCE_TRANSCRIPTION = 'reference/reference_transcription.txt'


def load_transcription(path):
    """Load the reference transcription file, returning {address: [bytes]}."""
    lines = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove trailing [?] markers
            line = line.replace('[?]', '').strip()
            parts = line.split(':')
            if len(parts) != 2:
                continue
            try:
                addr = int(parts[0].strip(), 16)
            except ValueError:
                continue
            hex_str = parts[1].strip()
            byte_vals = []
            for token in hex_str.split():
                try:
                    byte_vals.append(int(token, 16))
                except ValueError:
                    break
            if len(byte_vals) == 16:
                lines[addr] = byte_vals
    return lines


def hex_to_ascii(byte_val):
    """Convert a byte value to its ASCII printable character or '.'."""
    if 0x20 <= byte_val <= 0x7E:
        return chr(byte_val)
    return '.'


def validate_isp_id_code(transcription):
    """Check ISP ID code bytes against expected values."""
    print("\n=== ISP ID Code Validation ===")
    print(f"  {'Address':>8}  {'Expected':>8}  {'Actual':>8}  {'Match':>5}  ASCII")
    print(f"  {'-------':>8}  {'--------':>8}  {'------':>8}  {'-----':>5}  -----")

    match_count = 0
    total_count = 0
    for addr in sorted(ISP_ID_CODE.keys()):
        expected = ISP_ID_CODE[addr]
        row_addr = addr & 0xFFF0
        byte_offset = addr & 0x000F

        actual = None
        if row_addr in transcription:
            actual = transcription[row_addr][byte_offset]

        total_count += 1
        if actual is not None:
            match = actual == expected
            if match:
                match_count += 1
            status = "YES" if match else "[?]"
            ascii_char = hex_to_ascii(actual)
            print(f"  ${addr:05X}    {expected:02X}        {actual:02X}        "
                  f"{status}    '{ascii_char}'")
        else:
            print(f"  ${addr:05X}    {expected:02X}        --        N/A    (not in transcription)")

    print(f"\n  Result: {match_count}/{total_count} match")
    if match_count < total_count:
        print("  Note: $FFDF discrepancy may be due to different unit/revision")

    return match_count, total_count


def validate_ascii_cross_check(transcription):
    """Cross-validate hex values against expected ASCII patterns."""
    print("\n=== ASCII Cross-Check Validation ===")

    checks = [
        (0x0FF70, 0, 0x4E, 'N', "Row $0FF70 byte 0"),
        (0x0FFD0, 15, 0x48, 'H', "Row $0FFD0 byte 15 ($FFDF ID byte)"),
        (0x0FFE0, 3, 0x4B, 'K', "Row $0FFE0 byte 3 ($FFE3 ID byte)"),
        (0x0FFE0, 11, 0x30, '0', "Row $0FFE0 byte 11 ($FFEB ID byte)"),
        (0x0FFF0, 15, 0x5F, '_', "Row $0FFF0 byte 15 (reset vector byte 3)"),
    ]

    for row_addr, byte_idx, expected_val, expected_ascii, desc in checks:
        if row_addr in transcription:
            actual = transcription[row_addr][byte_idx]
            actual_ascii = hex_to_ascii(actual)
            match = actual == expected_val
            status = "OK" if match else "MISMATCH"
            print(f"  {desc}: {actual:02X}→'{actual_ascii}' "
                  f"(expected {expected_val:02X}→'{expected_ascii}') [{status}]")
        else:
            print(f"  {desc}: (row not in transcription)")


def analyze_patterns(transcription):
    """Report structural patterns in the transcription."""
    print("\n=== Pattern Analysis ===")

    # Check for repeating 4E FC 00 00 pattern
    repeat_count = 0
    for addr in sorted(transcription.keys()):
        row = transcription[addr]
        for i in range(0, 16, 4):
            if row[i:i + 4] == [0x4E, 0xFC, 0x00, 0x00]:
                repeat_count += 1

    print(f"  '4E FC 00 00' occurrences: {repeat_count} (×4 bytes each)")

    # Check 7-segment table
    if 0x10000 in transcription:
        row = transcription[0x10000]
        expected_7seg = [0xC0, 0xF9, 0xA4, 0xB0, 0x99, 0x92, 0x82, 0xD8, 0x80, 0x90]
        actual_7seg = row[:10]
        if actual_7seg == expected_7seg:
            print("  7-segment table at $10000: CORRECT (digits 0-9)")
        else:
            print("  7-segment table at $10000: MISMATCH")
            print(f"    Expected: {' '.join(f'{b:02X}' for b in expected_7seg)}")
            print(f"    Actual:   {' '.join(f'{b:02X}' for b in actual_7seg)}")

    # Check bit mask table
    if 0x10020 in transcription:
        row = transcription[0x10020]
        expected_masks = [0xFD, 0xFB, 0xF7, 0xEF, 0xDF, 0xBF, 0x7F, 0xFF]
        actual_masks = row[:8]
        if actual_masks == expected_masks:
            print("  Bit mask table at $10020: CORRECT (inverted single-bit masks)")
        else:
            print("  Bit mask table at $10020: MISMATCH")

    # Check reset vector
    if 0x0FFF0 in transcription:
        row = transcription[0x0FFF0]
        reset_bytes = row[12:16]  # $FFFC-$FFFF
        addr_20bit = (reset_bytes[2] & 0x0F) << 16 | reset_bytes[1] << 8 | reset_bytes[0]
        print(f"  Reset vector at $FFFC: "
              f"{reset_bytes[0]:02X} {reset_bytes[1]:02X} "
              f"{reset_bytes[2]:02X} {reset_bytes[3]:02X} "
              f"→ entry point ${addr_20bit:05X}")
        if 0x04000 <= addr_20bit <= 0x13FFF:
            print(f"    Entry point is in ROM range ($04000-$13FFF): VALID")
        else:
            print(f"    Entry point is OUTSIDE ROM range: INVALID")

    # Check ramp table monotonicity
    ramp_addrs = [0x10028, 0x10030, 0x10040, 0x10050, 0x10060]
    ramp_values = []
    for addr in sorted(transcription.keys()):
        if addr >= 0x10028:
            ramp_values.extend(transcription[addr])

    if ramp_values:
        # Check partial row from $10028 (bytes 8-15 of the $10020 row)
        if 0x10020 in transcription:
            partial = transcription[0x10020][8:]
            is_monotonic = all(partial[i] <= partial[i + 1]
                              for i in range(len(partial) - 1))
            print(f"  Ramp table at $10028: "
                  f"{'monotonic' if is_monotonic else 'NOT monotonic'} "
                  f"(values: {' '.join(f'{b:02X}' for b in partial)})")


def analyze_screenshot(image_path):
    """Analyze the reference screenshot image properties."""
    print("\n=== Screenshot Analysis ===")

    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: cannot read {image_path}")
        return

    h, w, c = img.shape
    print(f"  Image size: {w}×{h} ({c} channels)")

    # Detect green headers
    from calibrate_grid import detect_green_headers
    header_bottom, x_left, x_right, bars = detect_green_headers(img)
    print(f"  Header bottom y: {header_bottom}")
    print(f"  Dialog x-bounds: [{x_left}, {x_right}]")
    if len(bars) >= 3:
        print(f"  ADDRESS bar: [{bars[0][0]}, {bars[0][1]}]")
        print(f"  HEX bar:     [{bars[1][0]}, {bars[1][1]}]")
        print(f"  ASCII bar:   [{bars[2][0]}, {bars[2][1]}]")


def main():
    print("=" * 60)
    print("Reference Transcription Analysis")
    print("=" * 60)

    # Check files exist
    if not os.path.exists(REFERENCE_TRANSCRIPTION):
        print(f"Error: transcription not found: {REFERENCE_TRANSCRIPTION}")
        sys.exit(1)

    # Load transcription
    transcription = load_transcription(REFERENCE_TRANSCRIPTION)
    print(f"\nLoaded {len(transcription)} lines from {REFERENCE_TRANSCRIPTION}")
    if transcription:
        addrs = sorted(transcription.keys())
        total_bytes = len(transcription) * 16
        print(f"  Address range: ${addrs[0]:05X} – ${addrs[-1] + 0xF:05X}")
        print(f"  Total bytes: {total_bytes}")

    # Validate ISP ID code
    validate_isp_id_code(transcription)

    # ASCII cross-checks
    validate_ascii_cross_check(transcription)

    # Pattern analysis
    analyze_patterns(transcription)

    # Screenshot analysis (if available)
    if os.path.exists(REFERENCE_IMAGE):
        analyze_screenshot(REFERENCE_IMAGE)
    else:
        print(f"\n  Note: {REFERENCE_IMAGE} not found, skipping image analysis")

    print("\n" + "=" * 60)
    print("Analysis complete.")


if __name__ == '__main__':
    main()
