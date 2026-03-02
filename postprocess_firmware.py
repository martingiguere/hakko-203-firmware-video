#!/usr/bin/env python3
"""
Post-process extracted firmware for Hakko FM-203.

1. Load extraction from extracted_firmware.txt
2. Filter invalid/corrupt lines
3. Apply systematic error corrections
4. Merge extraction with reference data (reference wins)
5. FF-fill erased flash gaps (3-pass cascading)
6. Generate hex dump, source map, and binary files
7. Compute checksum (try multiple algorithms for $00D2F2FF)

Output files:
    firmware_merged.txt         — Human-readable hex dump (5,120 lines)
    firmware_merged_sources.json — Per-address source attribution
    hakko_fm203_full.bin        — Full 80 KB buffer ($00000-$13FFF)
    hakko_fm203.bin             — ROM-only 64 KB ($04000-$13FFF)
"""

import json
import os
import re
import struct
from collections import Counter, defaultdict

# Address range for R5F21258SNFP
BASE_ADDR = 0x00000
END_ADDR = 0x13FFF
BUFFER_SIZE = 0x14000   # 81920 bytes (80 KB)

# ROM region
ROM_START = 0x04000
ROM_END = 0x13FFF
ROM_SIZE = 0x10000      # 65536 bytes (64 KB)

# Expected checksum from Xeltek UI
EXPECTED_CHECKSUM = 0x00D2F2FF

# Known-bad addresses: confirmed OCR garbage from extraction.
# Excluded from merge even if present in extraction files.
BLOCKLIST = set()

# Confirmed erased flash regions — forced to all-FF after merge.
# Any non-FF data here is OCR noise.
# Start with SFR/RAM region which has no flash.
# More regions added after first extraction analysis.
FF_FORCED_REGIONS = [
    # SFR ($00000-$002FF) + reserved ($00300-$003FF) + RAM ($00400-$00FFF)
    # + unmapped gap ($01000-$03FFF) — no ROM here
    (0x00000, 0x03FFF),
]


def load_reference():
    """Load verified reference transcription as {addr_int: [hex_str_list]}."""
    from analyze_reference import load_transcription
    raw = load_transcription('reference/reference_transcription.txt')
    ref = {}
    for addr, byte_vals in raw.items():
        ref[addr] = [f'{b:02X}' for b in byte_vals]
    return ref


def load_extraction(filepath):
    """Load an extraction file.

    Returns dict of addr_int -> (hex_bytes_list, obs_count).
    obs_count = -1 for reference-sourced lines.
    """
    data = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split(':')
            if len(parts) < 2:
                continue

            try:
                addr = int(parts[0].strip(), 16)
            except ValueError:
                continue

            rest = parts[1].strip()
            obs = 1
            m = re.search(r'\[(\d+) obs\]', rest)
            if m:
                obs = int(m.group(1))
                rest = rest[:m.start()].strip()

            if '[REF]' in rest:
                obs = -1
                rest = rest.replace('[REF]', '').strip()

            hex_bytes = rest.split()
            if len(hex_bytes) == 16:
                valid = all(re.match(r'^[0-9A-Fa-f]{2}$', b) for b in hex_bytes)
                if valid:
                    data[addr] = ([b.upper() for b in hex_bytes], obs)

    return data


def is_valid_line(addr, hex_bytes, obs):
    """Check if a line of hex data appears valid.

    Filters out OCR artifacts like ASCII column bleed-through,
    repetitive non-FF patterns, and out-of-range addresses.
    """
    # Address must be 16-byte aligned
    if addr % 0x10 != 0:
        return False

    # Address must be in buffer range
    if addr < BASE_ADDR or addr > END_ADDR - 0x0F:
        return False

    # Reject lines with too many 7C bytes (ASCII column separator artifact)
    count_7c = sum(1 for b in hex_bytes if b == '7C')
    if count_7c >= 6:
        return False

    # Reject lines where most bytes are a suspicious repeated value
    byte_counts = Counter(hex_bytes)
    most_common_byte, most_common_count = byte_counts.most_common(1)[0]
    if most_common_byte not in ('FF', '00'):
        # Allow repetitive data in table regions
        if most_common_count >= 14:
            return False

    # Reject lines that look like ASCII column bleed-through
    ascii_range_count = sum(1 for b in hex_bytes
                           if 0x20 <= int(b, 16) <= 0x7E)
    # Don't filter vector table region by ASCII range
    if not (0x0FF00 <= addr <= 0x0FFF0):
        if ascii_range_count >= 14:
            return False

    return True


def apply_corrections(addr, hex_bytes, context=None):
    """Apply systematic error corrections.

    Returns corrected hex_bytes list.
    """
    corrected = list(hex_bytes)

    # In mostly-FF lines, CF should be FF (F->C OCR error)
    ff_count = sum(1 for b in corrected if b == 'FF')
    cf_count = sum(1 for b in corrected if b == 'CF')

    if ff_count >= 10 and cf_count >= 1:
        for i, b in enumerate(corrected):
            if b == 'CF':
                corrected[i] = 'FF'

    # OCR trailing byte corruption: ASCII column bleed-through
    # produces trailing 33/73/83 instead of FF in mostly-FF lines.
    ff_count = sum(1 for b in corrected if b == 'FF')
    if ff_count >= 14 and corrected[15] in ('33', '73', '83'):
        corrected[15] = 'FF'

    return corrected


def merge_and_vote(extraction, reference):
    """Merge extraction with reference data.

    Priority:
    1. Reference data (always wins)
    2. Extraction data weighted by observation count

    Returns dict of addr -> {bytes, source, confidence}.
    """
    final = {}

    all_addrs = set()
    all_addrs.update(extraction.keys())
    all_addrs.update(reference.keys())

    for addr in sorted(all_addrs):
        if addr < BASE_ADDR or addr > END_ADDR - 0x0F or addr % 0x10 != 0:
            continue
        if addr in BLOCKLIST:
            continue

        # Reference data always wins
        if addr in reference:
            final[addr] = {
                'bytes': list(reference[addr]),
                'source': 'reference',
                'confidence': 1.0,
            }
            continue

        if addr in extraction:
            hex_bytes, obs = extraction[addr]
            if is_valid_line(addr, hex_bytes, obs):
                corrected = apply_corrections(addr, hex_bytes)

                # Confidence based on observation count
                if obs >= 5:
                    confidence = 0.6
                elif obs >= 2:
                    confidence = 0.4
                else:
                    confidence = 0.2

                final[addr] = {
                    'bytes': corrected,
                    'source': 'extraction',
                    'confidence': confidence,
                }

    return final


def ff_fill_gaps(final_data):
    """Fill gaps where both immediate neighbors are all-FF.

    Pass 1: direct neighbor fill.
    Cascades until no more fills are possible.
    """
    expected = set(range(BASE_ADDR, END_ADDR + 1, 0x10))
    missing = sorted(expected - set(final_data.keys()))

    filled = 0
    for addr in missing:
        prev_addr = addr - 0x10
        next_addr = addr + 0x10

        prev_is_ff = (prev_addr in final_data and
                      all(b == 'FF' for b in final_data[prev_addr]['bytes']))
        next_is_ff = (next_addr in final_data and
                      all(b == 'FF' for b in final_data[next_addr]['bytes']))

        if prev_is_ff and next_is_ff:
            final_data[addr] = {
                'bytes': ['FF'] * 16,
                'source': 'ff-fill',
                'confidence': 0.95,
            }
            filled += 1

    # Cascade until no more fills
    total_filled = filled
    while filled > 0:
        prev_filled = filled
        filled = 0
        missing = sorted(expected - set(final_data.keys()))
        for addr in missing:
            prev_addr = addr - 0x10
            next_addr = addr + 0x10
            prev_is_ff = (prev_addr in final_data and
                          all(b == 'FF' for b in final_data[prev_addr]['bytes']))
            next_is_ff = (next_addr in final_data and
                          all(b == 'FF' for b in final_data[next_addr]['bytes']))
            if prev_is_ff and next_is_ff:
                final_data[addr] = {
                    'bytes': ['FF'] * 16,
                    'source': 'ff-fill',
                    'confidence': 0.95,
                }
                filled += 1
        total_filled += filled
        if filled == prev_filled:
            break

    print(f"FF-fill (neighbor): added {total_filled} erased flash lines")
    return total_filled


def ff_fill_region(final_data):
    """Extended FF-fill using region context.

    A gap is filled if the nearest known line on EACH side (within 32 lines)
    is all-FF. Handles cases where gaps break an immediate-neighbor chain
    but the surrounding region is clearly erased flash.
    """
    expected = set(range(BASE_ADDR, END_ADDR + 1, 0x10))
    missing = sorted(expected - set(final_data.keys()))
    filled = 0

    for addr in missing:
        # Look backward for nearest known line
        prev_ff = False
        for offset in range(1, 33):
            check_addr = addr - offset * 0x10
            if check_addr < BASE_ADDR:
                break
            if check_addr in final_data:
                prev_ff = all(b == 'FF' for b in final_data[check_addr]['bytes'])
                break

        # Look forward for nearest known line
        next_ff = False
        for offset in range(1, 33):
            check_addr = addr + offset * 0x10
            if check_addr > END_ADDR:
                break
            if check_addr in final_data:
                next_ff = all(b == 'FF' for b in final_data[check_addr]['bytes'])
                break

        if prev_ff and next_ff:
            final_data[addr] = {
                'bytes': ['FF'] * 16,
                'source': 'ff-fill-region',
                'confidence': 0.85,
            }
            filled += 1

    # Cascade
    total_filled = filled
    while filled > 0:
        prev = filled
        filled = 0
        missing = sorted(expected - set(final_data.keys()))
        for addr in missing:
            prev_ff = False
            for offset in range(1, 33):
                check_addr = addr - offset * 0x10
                if check_addr < BASE_ADDR:
                    break
                if check_addr in final_data:
                    prev_ff = all(b == 'FF' for b in final_data[check_addr]['bytes'])
                    break
            next_ff = False
            for offset in range(1, 33):
                check_addr = addr + offset * 0x10
                if check_addr > END_ADDR:
                    break
                if check_addr in final_data:
                    next_ff = all(b == 'FF' for b in final_data[check_addr]['bytes'])
                    break
            if prev_ff and next_ff:
                final_data[addr] = {
                    'bytes': ['FF'] * 16,
                    'source': 'ff-fill-region',
                    'confidence': 0.85,
                }
                filled += 1
        total_filled += filled
        if filled == prev:
            break

    print(f"FF-fill (region): added {total_filled} erased flash lines")
    return total_filled


def write_hex_dump(final_data, output_path):
    """Write the merged hex dump to a text file."""
    with open(output_path, 'w') as f:
        f.write("# Hakko FM-203 Firmware — Merged extraction\n")
        f.write(f"# {len(final_data)} address lines recovered\n")
        if final_data:
            f.write(f"# Address range: "
                    f"0x{min(final_data):05X} - 0x{max(final_data):05X}\n")
        f.write("# Sources: extraction + reference + FF-fill\n")
        f.write("#\n")

        for addr in sorted(final_data.keys()):
            data = final_data[addr]
            addr_str = f"{addr:05X}"
            hex_str = ' '.join(data['bytes'])
            source = data.get('source', '?')

            if source == 'reference':
                f.write(f"{addr_str}: {hex_str}  [REF]\n")
            else:
                f.write(f"{addr_str}: {hex_str}\n")

    print(f"Hex dump written to {output_path}")


def write_source_map(final_data, output_path):
    """Write per-address source map to JSON for the review tool."""
    source_map = {}
    for addr in sorted(final_data.keys()):
        addr_str = f"{addr:05X}"
        source_map[addr_str] = {
            'source': final_data[addr].get('source', 'unknown'),
            'confidence': final_data[addr].get('confidence', 0),
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(source_map, f, indent=2)

    print(f"Source map written to {output_path} ({len(source_map)} entries)")


def write_binary(final_data, full_path, rom_path):
    """Write firmware binary files.

    Args:
        final_data: merged firmware data
        full_path: output path for full 80 KB buffer ($00000-$13FFF)
        rom_path: output path for ROM-only 64 KB ($04000-$13FFF)

    Returns:
        (full_firmware, rom_firmware) bytearrays
    """
    # Full 80 KB buffer
    full_firmware = bytearray([0xFF] * BUFFER_SIZE)
    bytes_written = 0
    for addr, data in final_data.items():
        offset = addr - BASE_ADDR
        if 0 <= offset <= BUFFER_SIZE - 16:
            for i, b in enumerate(data['bytes']):
                full_firmware[offset + i] = int(b, 16)
                bytes_written += 1

    with open(full_path, 'wb') as f:
        f.write(full_firmware)

    print(f"\nFull binary written to {full_path} ({BUFFER_SIZE} bytes)")
    print(f"  Data bytes set: {bytes_written}")
    print(f"  Remaining 0xFF: {BUFFER_SIZE * 16 - bytes_written}")

    # ROM-only 64 KB
    rom_firmware = full_firmware[ROM_START - BASE_ADDR:
                                 ROM_START - BASE_ADDR + ROM_SIZE]

    with open(rom_path, 'wb') as f:
        f.write(rom_firmware)

    print(f"ROM binary written to {rom_path} ({ROM_SIZE} bytes)")

    return full_firmware, rom_firmware


def compute_checksum(firmware):
    """Try multiple checksum algorithms to match $00D2F2FF.

    The Xeltek UI shows checksum 00D2F2FF (32-bit).
    Try: byte sum, CRC-32, CRC-32/JAMCRC, word sum, etc.
    """
    print(f"\n=== Checksum Analysis (expected: {EXPECTED_CHECKSUM:08X}) ===")

    # 1. Simple byte sum (32-bit)
    byte_sum = sum(firmware) & 0xFFFFFFFF
    match = " *** MATCH ***" if byte_sum == EXPECTED_CHECKSUM else ""
    print(f"  Byte sum (32-bit):      {byte_sum:08X}{match}")

    # 2. Byte sum (no mask)
    byte_sum_full = sum(firmware)
    match = " *** MATCH ***" if byte_sum_full == EXPECTED_CHECKSUM else ""
    print(f"  Byte sum (unlimited):   {byte_sum_full:08X} "
          f"(dec: {byte_sum_full}){match}")

    # 3. CRC-32 (standard, zlib)
    import zlib
    crc32 = zlib.crc32(bytes(firmware)) & 0xFFFFFFFF
    match = " *** MATCH ***" if crc32 == EXPECTED_CHECKSUM else ""
    print(f"  CRC-32 (zlib):          {crc32:08X}{match}")

    # 4. CRC-32/JAMCRC (inverted CRC-32)
    jamcrc = crc32 ^ 0xFFFFFFFF
    match = " *** MATCH ***" if jamcrc == EXPECTED_CHECKSUM else ""
    print(f"  CRC-32/JAMCRC:          {jamcrc:08X}{match}")

    # 5. 16-bit word sum (little-endian, 32-bit result)
    word_sum = 0
    for i in range(0, len(firmware), 2):
        if i + 1 < len(firmware):
            word_sum += firmware[i] | (firmware[i + 1] << 8)
        else:
            word_sum += firmware[i]
    word_sum &= 0xFFFFFFFF
    match = " *** MATCH ***" if word_sum == EXPECTED_CHECKSUM else ""
    print(f"  16-bit word sum (LE):   {word_sum:08X}{match}")

    # 6. 16-bit word sum (big-endian)
    word_sum_be = 0
    for i in range(0, len(firmware), 2):
        if i + 1 < len(firmware):
            word_sum_be += (firmware[i] << 8) | firmware[i + 1]
        else:
            word_sum_be += firmware[i]
    word_sum_be &= 0xFFFFFFFF
    match = " *** MATCH ***" if word_sum_be == EXPECTED_CHECKSUM else ""
    print(f"  16-bit word sum (BE):   {word_sum_be:08X}{match}")

    # 7. 32-bit word sum (little-endian)
    dword_sum = 0
    for i in range(0, len(firmware), 4):
        val = 0
        for j in range(min(4, len(firmware) - i)):
            val |= firmware[i + j] << (8 * j)
        dword_sum += val
    dword_sum &= 0xFFFFFFFF
    match = " *** MATCH ***" if dword_sum == EXPECTED_CHECKSUM else ""
    print(f"  32-bit dword sum (LE):  {dword_sum:08X}{match}")

    # 8. CRC-16/CCITT
    crc16 = 0xFFFF
    for byte in firmware:
        crc16 ^= byte << 8
        for _ in range(8):
            if crc16 & 0x8000:
                crc16 = (crc16 << 1) ^ 0x1021
            else:
                crc16 <<= 1
            crc16 &= 0xFFFF
    print(f"  CRC-16/CCITT:           {crc16:04X}")

    # 9. Inverted byte sum
    inv_sum = (~sum(firmware) + 1) & 0xFFFFFFFF
    match = " *** MATCH ***" if inv_sum == EXPECTED_CHECKSUM else ""
    print(f"  Inverted byte sum:      {inv_sum:08X}{match}")

    return byte_sum


def compute_coverage(final_data):
    """Compute and report coverage statistics."""
    expected = set(range(BASE_ADDR, END_ADDR + 1, 0x10))
    found = set(final_data.keys()) & expected

    coverage = len(found) / len(expected) * 100
    print(f"\nCoverage: {len(found)}/{len(expected)} addresses ({coverage:.1f}%)")

    # Find gaps
    found_sorted = sorted(found)
    gaps = []
    if found_sorted:
        # Check start
        if found_sorted[0] > BASE_ADDR:
            gap_size = (found_sorted[0] - BASE_ADDR) // 0x10
            gaps.append((BASE_ADDR, found_sorted[0] - 0x10, gap_size))
        # Check interior
        for i in range(len(found_sorted) - 1):
            diff = found_sorted[i + 1] - found_sorted[i]
            if diff > 0x10:
                gap_size = (diff - 0x10) // 0x10
                gaps.append((found_sorted[i] + 0x10,
                            found_sorted[i + 1] - 0x10, gap_size))
        # Check end
        if found_sorted[-1] < END_ADDR - 0x0F:
            gap_size = (END_ADDR - 0x0F - found_sorted[-1]) // 0x10
            gaps.append((found_sorted[-1] + 0x10, END_ADDR - 0x0F, gap_size))

    if gaps:
        print(f"Gaps: {len(gaps)}")
        big_gaps = sorted(gaps, key=lambda x: -x[2])[:10]
        for start, end, size in big_gaps:
            print(f"  0x{start:05X}-0x{end:05X}: "
                  f"{size} lines ({size * 16} bytes)")

    # Source breakdown
    sources = Counter()
    for data in final_data.values():
        sources[data.get('source', 'unknown')] += 1
    print(f"\nSource breakdown:")
    for source, count in sources.most_common():
        print(f"  {source}: {count} lines")

    return coverage


def validate_firmware(final_data):
    """Run firmware validation checks."""
    print("\n=== Firmware Validation ===")

    # 1. Check ISP ID code bytes
    isp_checks = {
        0x0FFE3: 0x4B,  # 'K'
        0x0FFEB: 0x30,  # '0'
        0x0FFEF: 0x30,  # '0'
        0x0FFF3: 0x32,  # '2'
        0x0FFF7: 0x35,  # '5'
        0x0FFFB: 0x36,  # '6'
    }

    print("\nISP ID Code (6/7 expected to match):")
    id_match = 0
    id_total = 0
    for addr, expected in sorted(isp_checks.items()):
        row_addr = addr & 0xFFFF0
        byte_idx = addr & 0x0000F
        id_total += 1
        if row_addr in final_data:
            actual_str = final_data[row_addr]['bytes'][byte_idx]
            actual = int(actual_str, 16)
            match = actual == expected
            if match:
                id_match += 1
            status = "OK" if match else "MISMATCH"
            print(f"  ${addr:05X}: expected {expected:02X}, "
                  f"got {actual:02X} [{status}]")
        else:
            print(f"  ${addr:05X}: expected {expected:02X}, "
                  f"got -- [MISSING]")
    print(f"  Result: {id_match}/{id_total}")

    # 2. Check reset vector
    if 0x0FFF0 in final_data:
        row = final_data[0x0FFF0]['bytes']
        reset_bytes = [int(b, 16) for b in row[12:16]]  # $FFFC-$FFFF
        entry = ((reset_bytes[2] & 0x0F) << 16 |
                 reset_bytes[1] << 8 | reset_bytes[0])
        print(f"\nReset vector ($FFFC-$FFFF): "
              f"{reset_bytes[0]:02X} {reset_bytes[1]:02X} "
              f"{reset_bytes[2]:02X} {reset_bytes[3]:02X}")
        print(f"  Entry point: ${entry:05X}")
        if ROM_START <= entry <= ROM_END:
            print(f"  In ROM range: YES")
        else:
            print(f"  In ROM range: NO (WARNING)")

    # 3. Check reference data match
    ref_data = load_reference()
    ref_match = 0
    ref_total = 0
    ref_bytes_match = 0
    ref_bytes_total = 0
    for addr, expected_bytes in ref_data.items():
        if addr in final_data:
            ref_total += 1
            actual_bytes = final_data[addr]['bytes']
            line_ok = True
            for i in range(16):
                ref_bytes_total += 1
                if actual_bytes[i].upper() == expected_bytes[i].upper():
                    ref_bytes_match += 1
                else:
                    line_ok = False
            if line_ok:
                ref_match += 1

    print(f"\nReference match (256 ground-truth bytes):")
    if ref_total > 0:
        print(f"  Lines: {ref_match}/{ref_total} "
              f"({ref_match / ref_total * 100:.1f}%)")
        print(f"  Bytes: {ref_bytes_match}/{ref_bytes_total} "
              f"({ref_bytes_match / ref_bytes_total * 100:.1f}%)")


def main():
    print("=== Firmware Post-Processing (FM-203) ===\n")

    # Load reference
    reference = load_reference()
    print(f"Reference data: {len(reference)} lines")

    # Load extraction
    extraction_path = 'extracted_firmware.txt'
    if not os.path.exists(extraction_path):
        print(f"ERROR: {extraction_path} not found. "
              f"Run extract_pipeline.py first.")
        return

    raw_data = load_extraction(extraction_path)
    print(f"Extraction data: {len(raw_data)} lines")

    # Filter extraction
    valid_extraction = {}
    filtered = 0
    for addr, (hex_bytes, obs) in raw_data.items():
        if is_valid_line(addr, hex_bytes, obs):
            corrected = apply_corrections(addr, hex_bytes)
            valid_extraction[addr] = (corrected, obs)
        else:
            filtered += 1
    print(f"\nFiltered {filtered} invalid lines")
    print(f"Valid extraction lines: {len(valid_extraction)}")

    # Merge with reference
    print("\nMerging data sources...")
    final = merge_and_vote(valid_extraction, reference)
    print(f"Merged data (before FF-fill): {len(final)} addresses")

    # Pass 1: FF-fill neighbor gaps
    print("\nApplying FF-fill (pass 1 — neighbor)...")
    ff_fill_gaps(final)

    # Post-merge OCR correction: fix mostly-FF lines with artifacts
    ocr_fixed = 0
    for addr, data in list(final.items()):
        if data['source'] == 'reference':
            continue
        b = data['bytes']
        ff_count = sum(1 for x in b if x == 'FF')
        if ff_count >= 13:
            changed = False
            for i in range(16):
                if b[i] in ('33', '73', '83'):
                    b[i] = 'FF'
                    changed = True
                elif b[i] == 'CF':
                    b[i] = 'FF'
                    changed = True
            if changed:
                ocr_fixed += 1
    if ocr_fixed:
        print(f"\nPost-merge OCR correction: fixed {ocr_fixed} mostly-FF lines")
        print("Applying FF-fill (pass 2 — cascade)...")
        ff_fill_gaps(final)

    # Pass 3: Extended FF-fill with region context
    print("\nApplying FF-fill (pass 3 — region context)...")
    ff_fill_region(final)

    # Force confirmed erased flash regions to all-FF
    ff_forced = 0
    for region_start, region_end in FF_FORCED_REGIONS:
        for addr in range(region_start, region_end + 0x10, 0x10):
            if addr in final:
                if not all(b == 'FF' for b in final[addr]['bytes']):
                    final[addr]['bytes'] = ['FF'] * 16
                    final[addr]['source'] = 'ff-forced'
                    final[addr]['confidence'] = 1.0
                    ff_forced += 1
            else:
                final[addr] = {
                    'bytes': ['FF'] * 16,
                    'source': 'ff-forced',
                    'confidence': 1.0,
                }
                ff_forced += 1
    if ff_forced:
        regions_str = ', '.join(
            f'0x{s:05X}-0x{e:05X}' for s, e in FF_FORCED_REGIONS)
        print(f"\nForced {ff_forced} lines to all-FF in "
              f"erased regions ({regions_str})")

    print(f"\nFinal data: {len(final)} addresses")

    # Coverage
    coverage = compute_coverage(final)

    # Write outputs
    write_hex_dump(final, 'firmware_merged.txt')
    write_source_map(final, 'firmware_merged_sources.json')
    full_fw, rom_fw = write_binary(
        final, 'hakko_fm203_full.bin', 'hakko_fm203.bin')

    # Checksum (try on full buffer)
    compute_checksum(full_fw)

    # Also try checksum on ROM-only
    print(f"\n--- ROM-only ({ROM_SIZE} bytes) ---")
    compute_checksum(rom_fw)

    # Validation
    validate_firmware(final)

    print("\n" + "=" * 60)
    print("Post-processing complete.")


if __name__ == '__main__':
    main()
