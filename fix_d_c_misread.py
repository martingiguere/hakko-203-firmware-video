#!/usr/bin/env python3
"""
Fix C↔D OCR address confusion in crop_index.json.

The kNN classifier confuses hex digits C and D in address fields.  Because the
video scrolls monotonically through address space, a frame's neighbors (±10
frame numbers) reveal its true address range.  Any frame assigned to an address
containing C whose neighbors suggest it should be D (or vice versa) is a
misread.

The correction swaps 'C'↔'D' digits in the address to find the candidate
closest to the neighbor median.

Steps:
  1. Detect suspicious frames via neighbor-context heuristic
  2. Determine corrected addresses using C↔D digit swap + neighbor proximity
  3. Move frame data (readings, confidences, crop PNGs) to corrected addresses
  4. Recompute byte consensus → update extracted_firmware.txt
  5. Rebuild downstream files and reset review state
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

CROP_INDEX_PATH = os.path.join(PROJECT_ROOT, 'crops', 'crop_index.json')
EXTRACTED_FW_PATH = os.path.join(PROJECT_ROOT, 'extracted_firmware.txt')
REVIEW_STATE_PATH = os.path.join(PROJECT_ROOT, 'review_state.json')
MERGED_FW_PATH = os.path.join(PROJECT_ROOT, 'firmware_merged.txt')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
FRAME_MOVES_PATH = os.path.join(PROJECT_ROOT, 'frame_moves.json')
VENV_PYTHON = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')

NEIGHBOR_RADIUS = 10  # frames to check in each direction
DISTANCE_THRESHOLD = 0x80  # minimum distance from median to consider a swap


def load_crop_index():
    with open(CROP_INDEX_PATH) as f:
        return json.load(f)


def save_crop_index(crop_index):
    with open(CROP_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(crop_index, f, indent=2)
    print(f"  Saved {CROP_INDEX_PATH}")


def build_frame_to_addrs(crop_index):
    """Build reverse map: frame number → list of addresses that claim it."""
    mapping = {}
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses':
            continue
        for f in entry.get('frames', []):
            mapping.setdefault(f, []).append(addr)
    return mapping


def swap_candidates(addr):
    """Generate all addresses reachable by swapping one or more 'C'↔'D' digits."""
    c_positions = [i for i, ch in enumerate(addr) if ch == 'C']
    d_positions = [i for i, ch in enumerate(addr) if ch == 'D']

    candidates = set()
    swap_positions = c_positions + d_positions

    for r in range(1, len(swap_positions) + 1):
        for combo in combinations(swap_positions, r):
            trial = list(addr)
            for pos in combo:
                trial[pos] = 'D' if trial[pos] == 'C' else 'C'
            candidates.add(''.join(trial))

    candidates.discard(addr)  # don't include self
    return candidates


# ── Phase 2: Anchor-based monotonicity correction ────────────────────────────

ANCHOR_RADIUS = 500   # frames to look for anchor points
SWAP_THRESHOLD = 0x800  # minimum improvement to justify a swap (~half of $1000)


def build_anchor_trajectory(crop_index):
    """
    Build ground-truth frame→address trajectory from addresses with NO C or D
    digits (unambiguous anchors).  Returns sorted list of (frame, addr_value).
    """
    anchors = []
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses':
            continue
        if 'C' in addr or 'D' in addr:
            continue
        addr_val = int(addr, 16)
        for frame in entry.get('frames', []):
            anchors.append((frame, addr_val))
    anchors.sort()
    return anchors


def estimate_expected_address(frame, anchors, radius=ANCHOR_RADIUS):
    """
    Estimate expected address at a given frame using nearby anchor points.
    Uses inverse-distance weighted median of anchors within ±radius frames.
    Returns expected address value, or None if no anchors in range.
    """
    nearby = []
    weights = []
    for af, av in anchors:
        dist = abs(af - frame)
        if dist <= radius and dist > 0:
            nearby.append(av)
            weights.append(1.0 / dist)

    if not nearby:
        return None

    # Weighted median: sort by value, find weight midpoint
    paired = sorted(zip(nearby, weights))
    total_w = sum(weights)
    cumulative = 0.0
    for val, w in paired:
        cumulative += w
        if cumulative >= total_w / 2.0:
            return val
    return paired[-1][0]


def detect_monotonicity_moves(crop_index):
    """
    Phase 2: Use anchor-based interpolation to detect C↔D misreads in blocks
    where all neighbors share the same wrong address (defeating the ±10 heuristic).

    Returns list of (source_addr, frame, dest_addr) tuples.
    """
    anchors = build_anchor_trajectory(crop_index)
    if not anchors:
        print("  Phase 2: No anchor points found — skipping")
        return []

    print(f"  Phase 2: Built anchor trajectory with {len(anchors)} points")

    moves = []
    for addr in sorted(crop_index):
        if addr == 'ref_addresses':
            continue
        if 'C' not in addr and 'D' not in addr:
            continue

        entry = crop_index[addr]
        addr_val = int(addr, 16)

        for frame in entry.get('frames', []):
            expected = estimate_expected_address(frame, anchors)
            if expected is None:
                continue

            current_dist = abs(addr_val - expected)

            # Try all C↔D swap candidates
            candidates = swap_candidates(addr)
            best_swap = None
            best_dist = current_dist

            for cand in candidates:
                cand_val = int(cand, 16)
                dist = abs(cand_val - expected)
                swap_magnitude = abs(addr_val - cand_val)
                adaptive_threshold = max(swap_magnitude // 2, 0x80)
                if dist < best_dist and (current_dist - dist) >= adaptive_threshold:
                    best_dist = dist
                    best_swap = cand

            if best_swap is not None:
                moves.append((addr, frame, best_swap))

    print(f"  Phase 2: Detected {len(moves)} frames to move via monotonicity")

    # Summarize
    mapping = {}
    for src, frame, dest in moves:
        mapping.setdefault((src, dest), []).append(frame)
    print(f"  Phase 2: Across {len(mapping)} unique source→dest pairs")

    return moves


def deduplicate_moves(moves1, moves2):
    """
    Merge two move lists, deduplicating on (source_addr, frame).
    Phase 2 (moves2) wins on conflict.
    """
    by_key = {}
    for src, frame, dest in moves1:
        by_key[(src, frame)] = (src, frame, dest)
    for src, frame, dest in moves2:
        by_key[(src, frame)] = (src, frame, dest)
    return list(by_key.values())


# ── Phase 1: Neighbor-context heuristic ──────────────────────────────────────

def detect_and_plan_moves(crop_index):
    """
    For each address containing a C or D digit, check every frame's neighbor
    context.  If the frame is far from its ±10 neighbors' median, compute the
    best C↔D swap address closest to the neighbor median.

    Returns a list of (source_addr, frame, dest_addr) tuples.
    """
    frame_to_addrs = build_frame_to_addrs(crop_index)
    moves = []  # (source_addr, frame, dest_addr)

    for addr in sorted(crop_index):
        if addr == 'ref_addresses':
            continue
        # Only consider addresses that contain C or D
        if 'C' not in addr and 'D' not in addr:
            continue

        entry = crop_index[addr]
        addr_val = int(addr, 16)

        for frame in entry.get('frames', []):
            # Gather neighbor addresses within ±NEIGHBOR_RADIUS
            neighbor_vals = []
            for delta in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
                nf = frame + delta
                if nf == frame:
                    continue
                if nf in frame_to_addrs:
                    for a in frame_to_addrs[nf]:
                        neighbor_vals.append(int(a, 16))

            if not neighbor_vals:
                continue

            median_val = statistics.median(neighbor_vals)

            # Is the current address far from its neighbors?
            if abs(addr_val - median_val) <= DISTANCE_THRESHOLD:
                continue

            # Try all C↔D swap candidates
            candidates = swap_candidates(addr)
            best_swap = None
            best_dist = abs(addr_val - median_val)  # must beat original

            for cand in candidates:
                cand_val = int(cand, 16)
                dist = abs(cand_val - median_val)
                if dist < best_dist:
                    best_dist = dist
                    best_swap = cand

            if best_swap is not None:
                moves.append((addr, frame, best_swap))

    print(f"  Detected {len(moves)} suspicious frames to move")

    # Summarize by unique source→dest
    mapping = {}
    for src, frame, dest in moves:
        mapping.setdefault((src, dest), []).append(frame)
    print(f"  Across {len(mapping)} unique source→dest pairs")

    return moves


# ── Step 3: Move frame data ──────────────────────────────────────────────────

def execute_moves(crop_index, moves):
    """
    Move frames from source to destination address entries.
    Moves readings, confidences, crop PNGs.  Logs to frame_moves.json.
    """
    affected_src = set()
    affected_dst = set()
    frames_moved = 0
    crops_moved = 0
    crops_missing = 0
    entries_emptied = 0

    # Group moves by (source, dest) for efficiency
    grouped = {}
    for src, frame, dest in moves:
        grouped.setdefault((src, dest), []).append(frame)

    for (src_key, dst_key), frame_list in sorted(grouped.items()):
        affected_src.add(src_key)
        affected_dst.add(dst_key)

        src_entry = crop_index.get(src_key)
        if src_entry is None:
            continue

        # Create destination entry if needed
        if dst_key not in crop_index:
            crop_index[dst_key] = {
                'frames': [],
                'readings': {},
                'confidences': {},
            }
        dst_entry = crop_index[dst_key]

        for frame in frame_list:
            if frame not in src_entry.get('frames', []):
                continue  # already moved or not present

            frame_str = str(frame)

            # Move readings and confidences
            if frame_str in src_entry.get('readings', {}):
                dst_entry.setdefault('readings', {})[frame_str] = \
                    src_entry['readings'].pop(frame_str)
            if frame_str in src_entry.get('confidences', {}):
                dst_entry.setdefault('confidences', {})[frame_str] = \
                    src_entry['confidences'].pop(frame_str)

            # Add frame to destination
            if frame not in dst_entry['frames']:
                dst_entry['frames'].append(frame)

            # Remove frame from source
            src_entry['frames'].remove(frame)

            # Move crop PNG
            src_dir = os.path.join(CROPS_DIR, src_key.lower())
            dst_dir = os.path.join(CROPS_DIR, dst_key.lower())
            crop_name = f"frame_{frame:05d}.png"
            src_path = os.path.join(src_dir, crop_name)
            dst_path = os.path.join(dst_dir, crop_name)

            if os.path.exists(src_path):
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src_path, dst_path)
                crops_moved += 1
            else:
                crops_missing += 1

            frames_moved += 1

        # Sort destination frames
        dst_entry['frames'] = sorted(dst_entry['frames'])

        # Clean up empty source entries
        if not src_entry['frames']:
            del crop_index[src_key]
            entries_emptied += 1
            src_dir = os.path.join(CROPS_DIR, src_key.lower())
            if os.path.isdir(src_dir) and not os.listdir(src_dir):
                os.rmdir(src_dir)

    print(f"  Frames moved: {frames_moved}")
    print(f"  Crops moved: {crops_moved} (missing: {crops_missing})")
    print(f"  Source addresses affected: {len(affected_src)}")
    print(f"  Source entries emptied (removed): {entries_emptied}")
    print(f"  Destination addresses created/updated: {len(affected_dst)}")

    # Log moves to frame_moves.json
    log_frame_moves(moves)

    return affected_src, affected_dst


def log_frame_moves(moves):
    """Append move records to frame_moves.json ledger."""
    import datetime

    existing = {"moves": []}
    if os.path.exists(FRAME_MOVES_PATH):
        with open(FRAME_MOVES_PATH) as f:
            existing = json.load(f)

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for src, frame, dest in moves:
        existing["moves"].append({
            "frame": frame,
            "from_addr": src,
            "to_addr": dest,
            "timestamp": ts,
        })

    with open(FRAME_MOVES_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)
    print(f"  Logged {len(moves)} moves to {FRAME_MOVES_PATH}")


# ── Step 4: Recompute byte consensus → update extracted_firmware.txt ─────────

def weighted_majority_vote(readings, confidences):
    """
    For each byte position, do weighted majority voting across frames.
    Weight = per-byte kNN confidence.
    Returns list of 16 hex strings and total observation count.
    """
    if not readings:
        return None, 0

    result = []
    for byte_idx in range(16):
        votes = Counter()
        for frame_str, byte_list in readings.items():
            if byte_idx < len(byte_list):
                weight = 1.0
                if frame_str in confidences:
                    conf_list = confidences[frame_str]
                    if byte_idx < len(conf_list):
                        weight = conf_list[byte_idx]
                votes[byte_list[byte_idx]] += weight
        if votes:
            result.append(votes.most_common(1)[0][0])
        else:
            result.append('FF')

    obs_count = len(readings)
    return result, obs_count


def update_extracted_firmware(crop_index, affected_src, affected_dst):
    """
    Recompute byte consensus for all affected addresses and update
    extracted_firmware.txt.
    """
    all_affected = affected_src | affected_dst

    # Load existing extracted_firmware.txt
    lines_by_addr = {}
    header_lines = []
    with open(EXTRACTED_FW_PATH) as f:
        for line in f:
            stripped = line.rstrip('\n')
            if stripped.lstrip().startswith('#') or not stripped.strip():
                header_lines.append(stripped)
                continue
            parts = stripped.split(':')
            if len(parts) >= 2:
                addr_str = parts[0].strip().upper()
                lines_by_addr[addr_str] = stripped

    updated = 0
    added = 0
    for addr_key in sorted(all_affected):
        if addr_key in crop_index:
            entry = crop_index[addr_key]
            readings = entry.get('readings', {})
            confidences = entry.get('confidences', {})
            consensus, obs_count = weighted_majority_vote(readings, confidences)

            if consensus and obs_count > 0:
                hex_str = ' '.join(consensus)
                new_line = f"{addr_key}: {hex_str}  [{obs_count} obs]"
                if addr_key in lines_by_addr:
                    updated += 1
                else:
                    added += 1
                lines_by_addr[addr_key] = new_line
        else:
            # Entry was removed (emptied source) — check if it had [REF] data
            if addr_key in lines_by_addr:
                existing = lines_by_addr[addr_key]
                if '[REF]' not in existing:
                    del lines_by_addr[addr_key]

    with open(EXTRACTED_FW_PATH, 'w') as f:
        for h in header_lines:
            f.write(h + '\n')
        for addr_str in sorted(lines_by_addr.keys(),
                                key=lambda x: int(x, 16)):
            f.write(lines_by_addr[addr_str] + '\n')

    print(f"  Updated {updated} lines, added {added} lines in extracted_firmware.txt")


# ── Step 5: Rebuild downstream files ────────────────────────────────────────

def rebuild_downstream():
    """Run postprocess_firmware.py and precompute_gaps.py."""
    print("\n  Running postprocess_firmware.py...")
    result = subprocess.run(
        [VENV_PYTHON, 'postprocess_firmware.py'],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: postprocess_firmware.py failed:\n{result.stderr}")
        sys.exit(1)
    for line in result.stdout.strip().split('\n'):
        if any(kw in line for kw in ['Coverage', 'Final data', 'Hex dump',
                                      'Source map', 'binary', 'MATCH']):
            print(f"    {line}")

    print("\n  Running precompute_gaps.py...")
    result = subprocess.run(
        [VENV_PYTHON, 'firmware_review_tool/precompute_gaps.py'],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: precompute_gaps.py failed:\n{result.stderr}")
        sys.exit(1)
    for line in result.stdout.strip().split('\n'):
        if any(kw in line for kw in ['Missing', 'Addresses with', 'Unreachable',
                                      'Total context', 'Done']):
            print(f"    {line}")


# ── Step 6: Reset review state for affected addresses ────────────────────────

def reset_review_state(affected_src, affected_dst):
    """Reset review status for all affected addresses."""
    if not os.path.exists(REVIEW_STATE_PATH):
        print("  review_state.json not found — skipping")
        return

    with open(REVIEW_STATE_PATH) as f:
        state = json.load(f)

    if 'lines' not in state:
        print("  review_state.json has no 'lines' — skipping")
        return

    # Load current firmware_merged.txt for updated bytes
    merged_bytes = {}
    with open(MERGED_FW_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                addr_str = parts[0].strip().upper()
                rest = parts[1].strip()
                rest = re.sub(r'\[.*?\]', '', rest).strip()
                hex_bytes = rest.split()
                if len(hex_bytes) == 16:
                    merged_bytes[addr_str] = hex_bytes

    all_affected = affected_src | affected_dst
    reset_count = 0

    for addr_key in sorted(all_affected):
        if addr_key in state['lines']:
            if addr_key in merged_bytes:
                state['lines'][addr_key]['bytes'] = merged_bytes[addr_key]
                state['lines'][addr_key]['source'] = 'merged'
            state['lines'][addr_key]['status'] = 'unreviewed'
            state['lines'][addr_key]['edited_positions'] = []
            reset_count += 1
        elif addr_key in merged_bytes:
            state['lines'][addr_key] = {
                'status': 'unreviewed',
                'bytes': merged_bytes[addr_key],
                'source': 'merged',
                'edited_positions': [],
            }
            reset_count += 1

    with open(REVIEW_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print(f"  Reset {reset_count} addresses in review_state.json")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Fix C\u2194D OCR Address Confusion")
    print("=" * 60)

    # Phase 1: Neighbor-context detection
    print("\nPhase 1: Detect suspicious frames via neighbor context")
    crop_index = load_crop_index()
    moves_p1 = detect_and_plan_moves(crop_index)

    # Phase 2: Anchor-based monotonicity detection
    print("\nPhase 2: Detect suspicious frames via anchor monotonicity")
    moves_p2 = detect_monotonicity_moves(crop_index)

    # Merge and deduplicate
    moves = deduplicate_moves(moves_p1, moves_p2)
    print(f"\n  Combined: {len(moves)} moves ({len(moves_p1)} phase 1, {len(moves_p2)} phase 2, {len(moves_p1) + len(moves_p2) - len(moves)} deduped)")

    if not moves:
        print("\nNo suspicious frames found. Nothing to do.")
        return

    # Dry-run summary
    src_addrs = set(m[0] for m in moves)
    dst_addrs = set(m[2] for m in moves)
    print(f"\n  Will move frames from {len(src_addrs)} source addresses")
    print(f"  to {len(dst_addrs)} destination addresses")

    # Step 3: Execute moves
    print("\nStep 3: Move frames to corrected addresses")
    affected_src, affected_dst = execute_moves(crop_index, moves)

    # Save crop_index
    print("\nSaving updated crop_index.json...")
    save_crop_index(crop_index)

    # Step 4: Recompute consensus
    print("\nStep 4: Recompute byte consensus")
    update_extracted_firmware(crop_index, affected_src, affected_dst)

    # Step 5: Rebuild downstream
    print("\nStep 5: Rebuild downstream files")
    rebuild_downstream()

    # Step 6: Reset review state
    print("\nStep 6: Reset review state")
    reset_review_state(affected_src, affected_dst)

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Source addresses affected: {len(affected_src)}")
    print(f"  Destination addresses affected: {len(affected_dst)}")

    # Verification: check 0x0D050-0x0DF70 gap
    crop_index_check = load_crop_index()
    d_range = [k for k in crop_index_check
               if k.startswith('0D') and 0x0D050 <= int(k, 16) <= 0x0DF70]
    print(f"  0x0D050-0x0DF70 entries now in crop_index: {len(d_range)}")

    # Check merged firmware for those addresses
    merged_d_count = 0
    if os.path.exists(MERGED_FW_PATH):
        with open(MERGED_FW_PATH) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        addr = int(parts[0].strip(), 16)
                        if 0x0D050 <= addr <= 0x0DF70:
                            merged_d_count += 1
                    except ValueError:
                        pass
    print(f"  0x0D050-0x0DF70 lines in firmware_merged.txt: {merged_d_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
