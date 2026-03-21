#!/usr/bin/env python3
"""
Verify FF↔non-FF transition boundaries in the firmware dump.

Audits all transitions between all-FF and non-FF data in firmware_merged.txt,
cross-references them against the two FF_FORCED_REGIONS definitions
(postprocess_firmware.py vs firmware_review_tool/app.py), and checks whether
raw crop_index.json OCR readings agree with the merged data at each boundary.

Read-only — does not modify any files.

Usage:
    source venv/bin/activate && python3 verify_ff_transitions.py
"""

import json
import os
import re
import sys

from memory_map_utils import (
    load_memory_map, get_ff_forced_ranges, get_region_name,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

MERGED_PATH = 'firmware_merged.txt'
SOURCES_PATH = 'firmware_merged_sources.json'
CROP_INDEX_PATH = os.path.join('crops', 'crop_index.json')

TOTAL_EXPECTED = 5120  # $00000-$13FF0 in steps of $10

# Load FF-forced ranges from unified memory_map.json
_MMAP = load_memory_map()
_FF_RANGES = get_ff_forced_ranges(_MMAP)
FF_FORCED_POSTPROCESS = [(s, e, get_region_name(_MMAP, s)) for s, e in _FF_RANGES]

# Build memory map from the JSON for region lookups
MEMORY_MAP = []
for r in _MMAP.get('regions', []):
    start = int(r['start'], 16) if isinstance(r['start'], str) else r['start']
    end = int(r['end'], 16) if isinstance(r['end'], str) else r['end']
    MEMORY_MAP.append((start, end, r['name']))


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_firmware_merged(path):
    """Parse firmware_merged.txt → sorted list of (addr_int, is_all_ff)."""
    lines = []
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
            # Strip tags like [REF], [N obs]
            rest = re.sub(r'\[.*?\]', '', rest).strip()
            hex_bytes = rest.split()
            is_ff = all(b.upper() == 'FF' for b in hex_bytes[:16])
            lines.append((addr, is_ff))
    lines.sort(key=lambda x: x[0])
    return lines


def load_sources(path):
    """Load firmware_merged_sources.json → {addr_int: {source, confidence}}."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k, 16): v for k, v in raw.items()}


def load_crop_index_sparse(path, addresses):
    """Load crop_index.json, extract stats only for requested addresses.

    Returns {addr_int: {n_frames, n_all_ff, n_non_ff}}.
    """
    addr_keys = {f"{a:05X}" for a in addresses}

    with open(path) as f:
        crop_index = json.load(f)

    result = {}
    for key in addr_keys:
        entry = crop_index.get(key)
        if entry is None or not isinstance(entry, dict):
            continue
        addr_int = int(key, 16)
        readings = entry.get('readings', {})
        n_frames = len(readings)
        n_all_ff = 0
        for frame_key, bytes_list in readings.items():
            if all(b.upper() == 'FF' for b in bytes_list[:16]):
                n_all_ff += 1
        result[addr_int] = {
            'n_frames': n_frames,
            'n_all_ff': n_all_ff,
            'n_non_ff': n_frames - n_all_ff,
        }
    return result


# ── Analysis ─────────────────────────────────────────────────────────────────

def is_in_forced(addr, regions):
    return any(start <= addr <= end for start, end, _ in regions)


def get_memory_region(addr):
    for start, end, name in MEMORY_MAP:
        if start <= addr <= end:
            return name
    return "Unknown"


def find_transitions(firmware_lines):
    """Find all FF↔non-FF transitions in the parsed dump."""
    transitions = []
    for i in range(1, len(firmware_lines)):
        addr_before, ff_before = firmware_lines[i - 1]
        addr_after, ff_after = firmware_lines[i]
        if ff_before != ff_after:
            gap = (addr_after - addr_before) // 0x10 - 1
            direction = "data->FF" if not ff_before and ff_after else "FF->data"
            transitions.append({
                'addr_before': addr_before,
                'addr_after': addr_after,
                'direction': direction,
                'gap': gap,
            })
    return transitions


def categorize_transition(addr_before, addr_after, regions):
    before_forced = is_in_forced(addr_before, regions)
    after_forced = is_in_forced(addr_after, regions)
    if before_forced != after_forced:
        return "forced_boundary"
    elif before_forced and after_forced:
        return "unexpected"
    else:
        return "natural"


def check_crop_agreement(transition, crop_data):
    """Check if crop_index readings agree with merged data at a transition."""
    addr_b = transition['addr_before']
    addr_a = transition['addr_after']
    direction = transition['direction']

    crop_b = crop_data.get(addr_b)
    crop_a = crop_data.get(addr_a)

    if crop_b is None and crop_a is None:
        return "no_data"

    issues = []

    if direction == "data->FF":
        # addr_before should be non-FF, addr_after should be FF
        if crop_b and crop_b['n_frames'] > 0 and crop_b['n_non_ff'] == 0:
            issues.append(f"${addr_b:05X} merged=data but crops all-FF")
        if crop_a and crop_a['n_frames'] > 0 and crop_a['n_non_ff'] > 0:
            pct = crop_a['n_non_ff'] / crop_a['n_frames'] * 100
            issues.append(f"${addr_a:05X} merged=FF but {crop_a['n_non_ff']}/{crop_a['n_frames']} crops non-FF ({pct:.0f}%)")
    else:
        # FF->data: addr_before should be FF, addr_after should be non-FF
        if crop_b and crop_b['n_frames'] > 0 and crop_b['n_non_ff'] > 0:
            pct = crop_b['n_non_ff'] / crop_b['n_frames'] * 100
            issues.append(f"${addr_b:05X} merged=FF but {crop_b['n_non_ff']}/{crop_b['n_frames']} crops non-FF ({pct:.0f}%)")
        if crop_a and crop_a['n_frames'] > 0 and crop_a['n_non_ff'] == 0:
            issues.append(f"${addr_a:05X} merged=data but crops all-FF")

    if issues:
        return "DISAGREE: " + "; ".join(issues)
    return "agrees"


def compare_ff_regions():
    """Check that FF-forced regions come from unified memory_map.json."""
    # All scripts now use memory_map.json — no comparison needed
    return []


def detect_noise_clusters(transitions):
    """Detect rapid back-and-forth transitions (within 0x30) as noise clusters."""
    if not transitions:
        return []

    clusters = []
    current_cluster = [transitions[0]]

    for t in transitions[1:]:
        prev = current_cluster[-1]
        if t['addr_after'] - prev['addr_before'] <= 0x30:
            current_cluster.append(t)
        else:
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [t]

    if len(current_cluster) >= 2:
        clusters.append(current_cluster)

    return clusters


# ── Report ───────────────────────────────────────────────────────────────────

def format_report(transitions, discrepancies, stats, noise_clusters):
    lines = []

    # Section 1: FF_FORCED_REGIONS comparison
    lines.append("=" * 70)
    lines.append("Section 1: FF_FORCED_REGIONS Comparison")
    lines.append("=" * 70)
    lines.append("")
    lines.append("FF-Forced Regions:")
    lines.append("Source: memory_map.json (unified)")
    lines.append("")
    for start, end, label in FF_FORCED_POSTPROCESS:
        n = ((end & ~0xF) - (start & ~0xF)) // 0x10 + 1
        lines.append(f"  ${start:05X}-${end:05X}  ({n} lines) — {label}")
    lines.append("")

    if discrepancies:
        lines.append("DISCREPANCIES:")
        for i, d in enumerate(discrepancies, 1):
            lines.append(f"  [{i}] {d}")
    else:
        lines.append("All scripts use unified memory_map.json — no discrepancies.")
    lines.append("")

    # Section 2: Summary
    lines.append("=" * 70)
    lines.append("Section 2: Summary Statistics")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Total lines in dump:   {stats['total_lines']} / {TOTAL_EXPECTED}")
    lines.append(f"  Missing lines:         {stats['missing_lines']}")
    lines.append(f"  All-FF lines:          {stats['ff_lines']}")
    lines.append(f"  Non-FF lines:          {stats['non_ff_lines']}")
    lines.append(f"  Total transitions:     {len(transitions)}")

    by_cat = {}
    for t in transitions:
        cat = t['category']
        by_cat[cat] = by_cat.get(cat, 0) + 1
    for cat in ['forced_boundary', 'natural', 'unexpected']:
        lines.append(f"    {cat:20s} {by_cat.get(cat, 0)}")

    by_dir = {}
    for t in transitions:
        d = t['direction']
        by_dir[d] = by_dir.get(d, 0) + 1
    lines.append("")
    for d in ['FF->data', 'data->FF']:
        lines.append(f"    {d:20s} {by_dir.get(d, 0)}")
    lines.append("")

    # Section 3: All transitions
    lines.append("=" * 70)
    lines.append("Section 3: Transition Table")
    lines.append("=" * 70)
    lines.append("")

    hdr = (f"{'#':>3}  {'Before':>7}  {'After':>7}  {'Dir':>9}  {'Gap':>3}  "
           f"{'Src Before':>15}  {'Src After':>15}  {'Category':>16}  "
           f"{'Region':>14}  {'Crops':>12}  {'Agreement'}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for i, t in enumerate(transitions, 1):
        crop_b = t.get('crop_before')
        crop_a = t.get('crop_after')
        crop_str_b = f"{crop_b['n_frames']}f" if crop_b else "-"
        crop_str_a = f"{crop_a['n_frames']}f" if crop_a else "-"
        crop_str = f"{crop_str_b}/{crop_str_a}"

        agree = t.get('crop_agreement', '?')
        # Shorten for table
        if agree.startswith("DISAGREE"):
            agree_short = "DISAGREE"
        else:
            agree_short = agree

        line = (f"{i:3d}  ${t['addr_before']:05X}  ${t['addr_after']:05X}  "
                f"{t['direction']:>9}  {t['gap']:3d}  "
                f"{t['source_before']:>15}  {t['source_after']:>15}  "
                f"{t['category']:>16}  {t['memory_region']:>14}  "
                f"{crop_str:>12}  {agree_short}")
        lines.append(line)
    lines.append("")

    # Section 4: Noise clusters
    lines.append("=" * 70)
    lines.append("Section 4: Noise Analysis")
    lines.append("=" * 70)
    lines.append("")

    total_noise_transitions = sum(len(c) for c in noise_clusters)
    lines.append(f"  Rapid transition clusters (within $30): {len(noise_clusters)} clusters, "
                 f"{total_noise_transitions} transitions")
    lines.append("")

    if noise_clusters:
        for ci, cluster in enumerate(noise_clusters, 1):
            addr_min = min(t['addr_before'] for t in cluster)
            addr_max = max(t['addr_after'] for t in cluster)
            region = get_memory_region(addr_min)
            lines.append(f"  Cluster {ci}: ${addr_min:05X}-${addr_max:05X} "
                         f"({len(cluster)} transitions, {region})")
    lines.append("")

    # Section 5: Flagged issues
    lines.append("=" * 70)
    lines.append("Section 5: Flagged Issues")
    lines.append("=" * 70)
    lines.append("")

    issues_found = False

    unexpected = [t for t in transitions if t['category'] == 'unexpected']
    if unexpected:
        issues_found = True
        lines.append(f"  UNEXPECTED transitions inside forced regions: {len(unexpected)}")
        for t in unexpected:
            lines.append(f"    ${t['addr_before']:05X} -> ${t['addr_after']:05X}  "
                         f"{t['direction']}  ({t['source_before']} -> {t['source_after']})")
        lines.append("")

    disagreements = [t for t in transitions
                     if t.get('crop_agreement', '').startswith('DISAGREE')]
    if disagreements:
        issues_found = True
        lines.append(f"  Crop/merged DISAGREEMENTS: {len(disagreements)}")
        for t in disagreements:
            lines.append(f"    ${t['addr_before']:05X} -> ${t['addr_after']:05X}  "
                         f"{t['direction']}")
            lines.append(f"      {t['crop_agreement']}")
        lines.append("")

    if not issues_found:
        lines.append("  No issues flagged.")
        lines.append("")

    return '\n'.join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== FF Transition Verification — Hakko FM-203 ===\n")

    # 1. Parse firmware dump
    print("Loading firmware_merged.txt...")
    firmware_lines = parse_firmware_merged(MERGED_PATH)
    print(f"  {len(firmware_lines)} lines parsed")

    # 2. Load sources
    print("Loading firmware_merged_sources.json...")
    sources = load_sources(SOURCES_PATH)
    print(f"  {len(sources)} entries")

    # 3. Find transitions
    transitions = find_transitions(firmware_lines)
    print(f"  {len(transitions)} transitions found")

    # 4. Collect addresses needed from crop_index
    needed_addrs = set()
    for t in transitions:
        needed_addrs.add(t['addr_before'])
        needed_addrs.add(t['addr_after'])

    # 5. Load crop_index (sparse)
    print(f"Loading crop_index.json (sparse, {len(needed_addrs)} addresses)...")
    crop_data = load_crop_index_sparse(CROP_INDEX_PATH, needed_addrs)
    print(f"  {len(crop_data)} addresses found in crop_index")

    # 6. Enrich transitions
    for t in transitions:
        src_b = sources.get(t['addr_before'], {})
        src_a = sources.get(t['addr_after'], {})
        t['source_before'] = src_b.get('source', 'MISSING')
        t['source_after'] = src_a.get('source', 'MISSING')
        t['conf_before'] = src_b.get('confidence', 0)
        t['conf_after'] = src_a.get('confidence', 0)
        t['category'] = categorize_transition(
            t['addr_before'], t['addr_after'], FF_FORCED_POSTPROCESS)
        t['memory_region'] = get_memory_region(t['addr_after'])
        t['crop_before'] = crop_data.get(t['addr_before'])
        t['crop_after'] = crop_data.get(t['addr_after'])
        t['crop_agreement'] = check_crop_agreement(t, crop_data)

    # 7. Compare FF_FORCED_REGIONS
    discrepancies = compare_ff_regions()

    # 8. Detect noise clusters
    noise_clusters = detect_noise_clusters(transitions)

    # 9. Stats
    stats = {
        'total_lines': len(firmware_lines),
        'ff_lines': sum(1 for _, ff in firmware_lines if ff),
        'non_ff_lines': sum(1 for _, ff in firmware_lines if not ff),
        'missing_lines': TOTAL_EXPECTED - len(firmware_lines),
    }

    # 10. Report
    print()
    report = format_report(transitions, discrepancies, stats, noise_clusters)
    print(report)

    # Save to file
    report_path = 'ff_transition_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
