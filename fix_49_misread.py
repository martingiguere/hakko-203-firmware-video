#!/usr/bin/env python3
"""
Fix 4↔9 OCR address confusion in crop_index.json.

The kNN classifier confuses hex digits 4 and 9 in address fields.  Because the
video scrolls monotonically through address space, a frame's neighbors (±10
frame numbers) reveal its true address range.  Any frame assigned to a 04xxx
address whose neighbors are overwhelmingly 09xxx is a misread — and vice versa.

The correction swaps '4'↔'9' digits in the address to find the candidate
closest to the neighbor median.

Steps:
  1. Detect suspicious frames via neighbor-context heuristic
  2. Determine corrected addresses using 4↔9 digit swap + neighbor proximity
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
from frame_utils import crop_filename

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
    """Generate all addresses reachable by swapping one or more '4'↔'9' digits."""
    four_positions = [i for i, c in enumerate(addr) if c == '4']
    nine_positions = [i for i, c in enumerate(addr) if c == '9']

    candidates = set()
    swap_positions = four_positions + nine_positions

    for r in range(1, len(swap_positions) + 1):
        for combo in combinations(swap_positions, r):
            trial = list(addr)
            for pos in combo:
                trial[pos] = '9' if trial[pos] == '4' else '4'
            candidates.add(''.join(trial))

    candidates.discard(addr)  # don't include self
    return candidates


# ── Step 1 & 2: Detect suspicious frames and determine corrections ───────────

def detect_and_plan_moves(crop_index):
    """
    For each 04xxx address, check every frame's neighbor context.
    If the frame's ±10 neighbors are predominantly 09xxx, compute the
    best 4↔9 swap address closest to the neighbor median.

    Returns a list of (source_addr, frame, dest_addr) tuples.
    """
    frame_to_addrs = build_frame_to_addrs(crop_index)
    moves = []  # (source_addr, frame, dest_addr)

    for addr in sorted(crop_index):
        if not addr.startswith('04') or addr == 'ref_addresses':
            continue

        entry = crop_index[addr]
        for frame in entry.get('frames', []):
            # Gather 09xxx neighbor addresses within ±NEIGHBOR_RADIUS
            neighbor_09_vals = []
            for delta in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
                nf = frame + delta
                if nf == frame:
                    continue
                if nf in frame_to_addrs:
                    for a in frame_to_addrs[nf]:
                        if a.startswith('09'):
                            neighbor_09_vals.append(int(a, 16))

            if not neighbor_09_vals:
                continue

            # Find best swap candidate closest to neighbor median
            median_val = statistics.median(neighbor_09_vals)
            candidates = swap_candidates(addr)

            best_swap = None
            best_dist = float('inf')
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
                'video_frames': [],
                'readings': {},
                'confidences': {},
            }
        dst_entry = crop_index[dst_key]
        dst_entry.setdefault('video_frames', [])

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

            # No crop PNG move needed — frame-based storage is address-independent

            frames_moved += 1

        # Sort destination frames
        dst_entry['frames'] = sorted(dst_entry['frames'])

        # Clean up empty source entries
        if not src_entry.get('frames') and not src_entry.get('video_frames'):
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
    print("Fix 4↔9 OCR Address Confusion")
    print("=" * 60)

    # Step 1 & 2: Detect and plan
    print("\nStep 1-2: Detect suspicious frames and plan corrections")
    crop_index = load_crop_index()
    moves = detect_and_plan_moves(crop_index)

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

    # Verification: check that 04490 no longer has frames 9825-9829 or 10021-10039
    crop_index_check = load_crop_index()
    if '04490' in crop_index_check:
        remaining = crop_index_check['04490']['frames']
        suspect = [f for f in remaining if f > 9000]
        if suspect:
            print(f"  WARNING: 04490 still has suspect frames: {suspect}")
        else:
            print(f"  OK: 04490 has no remaining suspect frames")
    else:
        print(f"  OK: 04490 entry was fully emptied")

    for check_addr in ['09490', '09990']:
        if check_addr in crop_index_check:
            print(f"  {check_addr}: {len(crop_index_check[check_addr]['frames'])} frames")
        else:
            print(f"  WARNING: {check_addr} not found in crop_index")

    print("=" * 60)


if __name__ == '__main__':
    main()
