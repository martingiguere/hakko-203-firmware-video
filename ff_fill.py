#!/usr/bin/env python3
"""
FF-fill and FF-forced override for Hakko FM-203 firmware dump.

Reads memory_map.json to determine which regions are ff-forced, then:
  1. Forces all addresses in ff-forced regions to all-FF
  2. Optionally (--heuristic): applies neighbor-based and region-context
     FF-fill for remaining gaps in ROM

Runs at the end of the pipeline, after postprocess_firmware.py has produced
firmware_merged.txt without any FF-fill/forcing.

Input:  firmware_merged.txt, firmware_merged_sources.json
Output: firmware_merged.txt, firmware_merged_sources.json,
        hakko_fm203_full.bin, hakko_fm203.bin (overwritten in place)
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

from memory_map_utils import (
    load_memory_map, get_ff_forced_ranges, get_buffer_range,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

MERGED_PATH = 'firmware_merged.txt'
SOURCES_PATH = 'firmware_merged_sources.json'
FULL_BIN_PATH = 'hakko_fm203_full.bin'
ROM_BIN_PATH = 'hakko_fm203.bin'

ROM_START = 0x04000
ROM_END = 0x13FFF
ROM_SIZE = 0x10000


# ── Load / Save ─────────────────────────────────────────────────────────────

def load_merged(path):
    """Load firmware_merged.txt → {addr_int: {bytes, source, confidence}}."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':')
            if len(parts) < 2:
                continue
            try:
                addr = int(parts[0].strip(), 16)
            except ValueError:
                continue
            rest = parts[1].strip()
            is_ref = '[REF]' in rest
            rest = re.sub(r'\[.*?\]', '', rest).strip()
            hex_bytes = rest.split()[:16]
            data[addr] = {
                'bytes': hex_bytes,
                'source': 'reference' if is_ref else 'extraction',
                'confidence': 1.0 if is_ref else 0.5,
            }
    return data


def load_sources(path):
    """Load firmware_merged_sources.json → {addr_int: {source, confidence}}."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k, 16): v for k, v in raw.items()}


def apply_sources(data, sources):
    """Overlay source metadata from the JSON onto loaded data."""
    for addr, entry in data.items():
        if addr in sources:
            entry['source'] = sources[addr].get('source', entry['source'])
            entry['confidence'] = sources[addr].get('confidence', entry['confidence'])


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
    """Write per-address source map to JSON."""
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
    """Write firmware binary files."""
    buf_start = 0x00000
    buf_size = 0x14000

    full_firmware = bytearray([0xFF] * buf_size)
    for addr, data in final_data.items():
        offset = addr - buf_start
        if 0 <= offset <= buf_size - 16:
            for i, b in enumerate(data['bytes']):
                full_firmware[offset + i] = int(b, 16)

    with open(full_path, 'wb') as f:
        f.write(full_firmware)
    print(f"Full binary written to {full_path} ({buf_size} bytes)")

    rom_firmware = full_firmware[ROM_START - buf_start:
                                 ROM_START - buf_start + ROM_SIZE]
    with open(rom_path, 'wb') as f:
        f.write(rom_firmware)
    print(f"ROM binary written to {rom_path} ({ROM_SIZE} bytes)")


# ── FF-Forced Override ───────────────────────────────────────────────────────

def ff_force(final_data, ff_ranges):
    """Force all addresses in ff-forced ranges to all-FF.

    Creates entries for missing addresses, overwrites non-FF entries.
    """
    forced = 0
    for region_start, region_end in ff_ranges:
        for addr in range(region_start & ~0xF, (region_end & ~0xF) + 1, 0x10):
            if addr in final_data:
                if not all(b == 'FF' for b in final_data[addr]['bytes']):
                    final_data[addr]['bytes'] = ['FF'] * 16
                    final_data[addr]['source'] = 'ff-forced'
                    final_data[addr]['confidence'] = 1.0
                    forced += 1
            else:
                final_data[addr] = {
                    'bytes': ['FF'] * 16,
                    'source': 'ff-forced',
                    'confidence': 1.0,
                }
                forced += 1

    if forced:
        regions_str = ', '.join(
            f'${s:05X}-${e:05X}' for s, e in ff_ranges)
        print(f"FF-forced: {forced} lines overridden ({regions_str})")
    else:
        print("FF-forced: no changes needed")
    return forced


# ── Heuristic FF-Fill ────────────────────────────────────────────────────────

def ff_fill_neighbor(final_data, base_addr, end_addr):
    """Fill gaps where both immediate neighbors are all-FF. Cascading."""
    expected = set(range(base_addr, (end_addr & ~0xF) + 1, 0x10))

    total_filled = 0
    filled = 1  # seed the loop
    while filled > 0:
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

    print(f"FF-fill (neighbor): added {total_filled} lines")
    return total_filled


def ff_fill_region(final_data, base_addr, end_addr):
    """Fill gaps where nearest known line on each side (within 32 lines) is all-FF."""
    expected = set(range(base_addr, (end_addr & ~0xF) + 1, 0x10))

    total_filled = 0
    filled = 1
    while filled > 0:
        filled = 0
        missing = sorted(expected - set(final_data.keys()))
        for addr in missing:
            prev_ff = False
            for offset in range(1, 33):
                check = addr - offset * 0x10
                if check < base_addr:
                    break
                if check in final_data:
                    prev_ff = all(b == 'FF' for b in final_data[check]['bytes'])
                    break

            next_ff = False
            for offset in range(1, 33):
                check = addr + offset * 0x10
                if check > end_addr:
                    break
                if check in final_data:
                    next_ff = all(b == 'FF' for b in final_data[check]['bytes'])
                    break

            if prev_ff and next_ff:
                final_data[addr] = {
                    'bytes': ['FF'] * 16,
                    'source': 'ff-fill-region',
                    'confidence': 0.85,
                }
                filled += 1
        total_filled += filled

    print(f"FF-fill (region context): added {total_filled} lines")
    return total_filled


# ── OCR artifact cleanup ────────────────────────────────────────────────────

def fix_mostly_ff_artifacts(final_data):
    """Fix OCR artifacts in mostly-FF lines (CF→FF, trailing 33/73/83→FF)."""
    fixed = 0
    for addr, data in final_data.items():
        if data['source'] == 'reference':
            continue
        b = data['bytes']
        ff_count = sum(1 for x in b if x == 'FF')
        if ff_count >= 13:
            changed = False
            for i in range(16):
                if b[i] in ('33', '73', '83', 'CF'):
                    b[i] = 'FF'
                    changed = True
            if changed:
                fixed += 1
    if fixed:
        print(f"OCR artifact cleanup: fixed {fixed} mostly-FF lines")
    return fixed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FF-fill and FF-forced override for firmware dump')
    parser.add_argument('--heuristic', action='store_true',
                        help='Apply heuristic FF-fill (neighbor + region context)')
    args = parser.parse_args()

    print("=== FF-Fill & FF-Forced Override (FM-203) ===\n")

    # Load memory map
    mmap = load_memory_map()
    ff_ranges = get_ff_forced_ranges(mmap)
    buf_start, buf_end = get_buffer_range(mmap)

    print(f"Memory map: {mmap['device']}")
    print(f"Buffer: ${buf_start:05X}-${buf_end:05X}")
    print(f"FF-forced ranges: {len(ff_ranges)}")
    for s, e in ff_ranges:
        print(f"  ${s:05X}-${e:05X}")

    # Load merged data
    print(f"\nLoading {MERGED_PATH}...")
    final = load_merged(MERGED_PATH)
    sources = load_sources(SOURCES_PATH)
    apply_sources(final, sources)
    print(f"  {len(final)} lines loaded")

    # Step 1: FF-forced override
    print(f"\nApplying FF-forced override...")
    ff_force(final, ff_ranges)

    # Step 2: Heuristic FF-fill (optional)
    if args.heuristic:
        print(f"\nApplying heuristic FF-fill...")

        # OCR artifact cleanup first (enables more neighbor fills)
        fix_mostly_ff_artifacts(final)

        # Neighbor-based fill (cascading)
        ff_fill_neighbor(final, buf_start, buf_end)

        # Region-context fill
        ff_fill_region(final, buf_start, buf_end)

    # Coverage stats
    expected = set(range(buf_start, (buf_end & ~0xF) + 1, 0x10))
    found = set(final.keys()) & expected
    print(f"\nFinal: {len(final)} lines, coverage {len(found)}/{len(expected)} "
          f"({len(found) / len(expected) * 100:.1f}%)")

    source_counts = Counter(d.get('source', '?') for d in final.values())
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    # Write outputs
    print()
    write_hex_dump(final, MERGED_PATH)
    write_source_map(final, SOURCES_PATH)
    write_binary(final, FULL_BIN_PATH, ROM_BIN_PATH)

    print("\nDone.")


if __name__ == '__main__':
    main()
