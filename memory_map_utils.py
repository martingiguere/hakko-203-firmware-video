#!/usr/bin/env python3
"""
Loader and query helpers for memory_map.json.

Provides a single source of truth for the R5F21258SNFP memory layout,
replacing hardcoded FF_FORCED_REGIONS and address range constants
scattered across multiple scripts.
"""

import json
import os

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'memory_map.json')

_cached_mmap = None
_cached_path = None


def load_memory_map(path=None):
    """Load memory_map.json and return the parsed dict.

    Caches the result — subsequent calls with the same path return
    the cached copy.
    """
    global _cached_mmap, _cached_path
    if path is None:
        path = _DEFAULT_PATH
    if _cached_mmap is not None and _cached_path == path:
        return _cached_mmap
    with open(path) as f:
        _cached_mmap = json.load(f)
    _cached_path = path
    return _cached_mmap


def _parse_addr(val):
    """Parse a hex address string like '0x02400' to int."""
    if isinstance(val, int):
        return val
    return int(val, 16)


def get_ff_forced_ranges(mmap=None):
    """Return list of (start, end) tuples for all ff-forced addresses.

    This is the union of:
      - regions with type "ff-forced"
      - ff_forced_overrides (ROM sub-ranges known to be erased)
    """
    if mmap is None:
        mmap = load_memory_map()
    ranges = []
    for r in mmap.get('regions', []):
        if r['type'] == 'ff-forced':
            ranges.append((_parse_addr(r['start']), _parse_addr(r['end'])))
    for o in mmap.get('ff_forced_overrides', []):
        ranges.append((_parse_addr(o['start']), _parse_addr(o['end'])))
    return sorted(ranges)


def get_rom_ranges(mmap=None):
    """Return list of (start, end) tuples for ROM regions.

    Excludes any sub-ranges covered by ff_forced_overrides.
    """
    if mmap is None:
        mmap = load_memory_map()
    overrides = [((_parse_addr(o['start']), _parse_addr(o['end'])))
                 for o in mmap.get('ff_forced_overrides', [])]
    ranges = []
    for r in mmap.get('regions', []):
        if r['type'] != 'rom':
            continue
        start = _parse_addr(r['start'])
        end = _parse_addr(r['end'])
        # Subtract any override ranges
        remaining = [(start, end)]
        for ov_start, ov_end in overrides:
            new_remaining = []
            for rs, re_ in remaining:
                if ov_end < rs or ov_start > re_:
                    new_remaining.append((rs, re_))
                else:
                    if rs < ov_start:
                        new_remaining.append((rs, ov_start - 1))
                    if re_ > ov_end:
                        new_remaining.append((ov_end + 1, re_))
            remaining = new_remaining
        ranges.extend(remaining)
    return sorted(ranges)


def get_data_ranges(mmap=None):
    """Return list of (start, end) tuples for data regions (e.g. Data Flash A)."""
    if mmap is None:
        mmap = load_memory_map()
    ranges = []
    for r in mmap.get('regions', []):
        if r['type'] == 'data':
            ranges.append((_parse_addr(r['start']), _parse_addr(r['end'])))
    return sorted(ranges)


def get_region_type(mmap, addr):
    """Return the region type for an address, or None if out of range.

    Checks ff_forced_overrides first (they take precedence over base region type).
    """
    for o in mmap.get('ff_forced_overrides', []):
        if _parse_addr(o['start']) <= addr <= _parse_addr(o['end']):
            return 'ff-forced'
    for r in mmap.get('regions', []):
        if _parse_addr(r['start']) <= addr <= _parse_addr(r['end']):
            return r['type']
    return None


def get_region_name(mmap, addr):
    """Return the region name for an address, or 'Unknown'."""
    for o in mmap.get('ff_forced_overrides', []):
        if _parse_addr(o['start']) <= addr <= _parse_addr(o['end']):
            return o['name']
    for r in mmap.get('regions', []):
        if _parse_addr(r['start']) <= addr <= _parse_addr(r['end']):
            return r['name']
    return 'Unknown'


def is_ff_forced(mmap, addr):
    """Check if an address falls in an ff-forced range."""
    return get_region_type(mmap, addr) == 'ff-forced'


def is_rom(mmap, addr):
    """Check if an address falls in a ROM range (not overridden to ff-forced)."""
    return get_region_type(mmap, addr) == 'rom'


def is_data(mmap, addr):
    """Check if an address falls in a data range."""
    return get_region_type(mmap, addr) == 'data'


def get_buffer_range(mmap=None):
    """Return (start, end) for the full buffer."""
    if mmap is None:
        mmap = load_memory_map()
    return (_parse_addr(mmap['buffer_start']), _parse_addr(mmap['buffer_end']))


def load_accepted_addresses(review_state_path='review_state.json'):
    """Load accepted/edited address keys and their byte data from review_state.json.

    Accepted addresses are treated as manually verified ground truth.
    Pipeline scripts must not overwrite or move frames at these addresses.

    Returns:
        accepted_addrs: set of 5-char uppercase hex address strings
        accepted_bytes: dict of addr_key -> list of 16 hex byte strings
    """
    accepted_addrs = set()
    accepted_bytes = {}

    if not os.path.exists(review_state_path):
        return accepted_addrs, accepted_bytes

    with open(review_state_path) as f:
        state = json.load(f)

    for addr_key, line in state.get('lines', {}).items():
        if line.get('status') in ('accepted', 'edited'):
            accepted_addrs.add(addr_key.upper())
            byte_data = line.get('bytes', [])
            if byte_data and '--' not in byte_data:
                accepted_bytes[addr_key.upper()] = byte_data

    return accepted_addrs, accepted_bytes


if __name__ == '__main__':
    mmap = load_memory_map()
    print(f"Device: {mmap['device']}")
    buf_start, buf_end = get_buffer_range(mmap)
    print(f"Buffer: ${buf_start:05X}-${buf_end:05X} ({mmap['buffer_size']} bytes)")
    print(f"\nFF-forced ranges:")
    for start, end in get_ff_forced_ranges(mmap):
        n = ((end & ~0xF) - (start & ~0xF)) // 0x10 + 1
        name = get_region_name(mmap, start)
        print(f"  ${start:05X}-${end:05X}  ({n} lines) — {name}")
    print(f"\nROM ranges:")
    for start, end in get_rom_ranges(mmap):
        n = ((end & ~0xF) - (start & ~0xF)) // 0x10 + 1
        name = get_region_name(mmap, start)
        print(f"  ${start:05X}-${end:05X}  ({n} lines) — {name}")
    print(f"\nData ranges:")
    for start, end in get_data_ranges(mmap):
        n = ((end & ~0xF) - (start & ~0xF)) // 0x10 + 1
        name = get_region_name(mmap, start)
        print(f"  ${start:05X}-${end:05X}  ({n} lines) — {name}")
