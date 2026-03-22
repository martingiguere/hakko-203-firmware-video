#!/usr/bin/env python3
"""
Strategy 7: Full-Video Gap Recovery

Scans full_video.mp4 for addresses missing from the firmware dump.
The pre-extracted frames/ directory has only 20,070 frames, but the video
has ~93,093 frames total. Many gap addresses are visible in un-extracted frames.

For each gap region:
  1. Estimate which video frames likely show those addresses
  2. Read frames from full_video.mp4 using OpenCV (in memory, no PNG saves)
  3. Run the kNN classifier on each unique frame
  4. Save row crops for recovered addresses
  5. Update crop_index.json and extracted_firmware.txt
  6. Rebuild downstream files
"""

import cv2
import json
import numpy as np
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

sys.path.insert(0, PROJECT_ROOT)
from extract_pipeline import FastKNNClassifier, process_frame, is_frame_different
from frame_utils import video_frame_key, crop_filename

CROP_INDEX_PATH = os.path.join(PROJECT_ROOT, 'crops', 'crop_index.json')
EXTRACTED_FW_PATH = os.path.join(PROJECT_ROOT, 'extracted_firmware.txt')
REVIEW_STATE_PATH = os.path.join(PROJECT_ROOT, 'review_state.json')
MERGED_FW_PATH = os.path.join(PROJECT_ROOT, 'firmware_merged.txt')
CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
VIDEO_PATH = os.path.join(PROJECT_ROOT, 'full_video.mp4')
CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, 'fast_knn_classifier.npz')
VENV_PYTHON = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')
FRAME_MOVES_PATH = os.path.join(PROJECT_ROOT, 'frame_moves.json')
FRAME_ASSIGNMENTS_PATH = os.path.join(PROJECT_ROOT, 'frame_assignments.json')

# Frame mapping: video_frame = extracted_frame + 24629
EXTRACTED_TO_VIDEO_OFFSET = 24629

# ROM range
ROM_MIN = 0x04000
ROM_MAX = 0x13FFF


def load_crop_index():
    with open(CROP_INDEX_PATH) as f:
        return json.load(f)


def save_crop_index(crop_index):
    tmp_path = CROP_INDEX_PATH + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(crop_index, f, indent=2)
    os.replace(tmp_path, CROP_INDEX_PATH)
    print(f"  Saved {CROP_INDEX_PATH}")


def log_gap_recovery(recovered_addrs):
    """Log recovered addresses to frame_moves.json for precompute replay."""
    import datetime
    data = {"moves": []}
    if os.path.exists(FRAME_MOVES_PATH):
        with open(FRAME_MOVES_PATH) as f:
            data = json.load(f)

    timestamp = datetime.datetime.now().isoformat()
    for addr_hex, video_frames in recovered_addrs.items():
        for vf in video_frames:
            data["moves"].append({
                "frame": f"v{vf}",
                "from_addr": "__gap__",
                "to_addr": addr_hex,
                "strategy": "gap_recovery",
                "timestamp": timestamp,
            })

    with open(FRAME_MOVES_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    total_new = sum(len(vfs) for vfs in recovered_addrs.values())
    print(f"  Logged {total_new} gap recovery entries to {FRAME_MOVES_PATH}")


def update_frame_assignments(recovered_addrs, crop_index):
    """Append video frame assignments to frame_assignments.json."""
    if not os.path.exists(FRAME_ASSIGNMENTS_PATH):
        return
    with open(FRAME_ASSIGNMENTS_PATH) as f:
        assignments = json.load(f)

    added = 0
    for addr_hex, video_frames in recovered_addrs.items():
        addr_int = int(addr_hex, 16)
        entry = crop_index.get(addr_hex, {})
        for vf in video_frames:
            vf_key = f"v{vf}"
            # Video frames use 'v'-prefixed keys in assignments
            if vf_key not in assignments:
                assignments[vf_key] = []
            # Get row_y from crop_index if available
            conf_list = entry.get('confidences', {}).get(vf_key, [])
            avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.5
            assignments[vf_key].append({
                'addr': addr_int, 'row_y': 0, 'conf': avg_conf
            })
            added += 1

    with open(FRAME_ASSIGNMENTS_PATH, 'w') as f:
        json.dump(assignments, f)
    print(f"  Appended {added} video frame assignments to {FRAME_ASSIGNMENTS_PATH}")


def find_missing_addresses():
    """Find ROM addresses missing from firmware_merged.txt."""
    covered = set()
    with open(MERGED_FW_PATH) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    addr = int(parts[0].strip(), 16)
                    covered.add(addr)
                except ValueError:
                    pass

    missing = []
    for a in range(ROM_MIN, ROM_MAX + 1, 0x10):
        if a not in covered:
            missing.append(a)
    return missing


def build_frame_address_map(crop_index):
    """Build sorted list of (video_frame, addr_int) from crop_index.

    Converts extracted frame numbers to video frame numbers.
    Video frames are used as-is (already in video coordinates).
    """
    points = []
    for addr_str, entry in crop_index.items():
        if addr_str == 'ref_addresses':
            continue
        try:
            addr_int = int(addr_str, 16)
        except ValueError:
            continue
        for frame in entry.get('frames', []):
            # Convert extracted frame → video frame
            vf = frame + EXTRACTED_TO_VIDEO_OFFSET
            points.append((vf, addr_int))
        for vf in entry.get('video_frames', []):
            points.append((vf, addr_int))
    points.sort()
    return points


def estimate_video_frame_for_address(target_addr, frame_addr_map):
    """Estimate which video frame number shows a given address.

    Uses linear interpolation between surrounding known points.
    Returns (estimated_frame, confidence_window) or (None, None).
    """
    # Binary search for surrounding points
    lo, hi = None, None
    for vf, addr in frame_addr_map:
        if addr <= target_addr:
            lo = (vf, addr)
        if addr >= target_addr and hi is None:
            hi = (vf, addr)
            break

    if lo is None and hi is None:
        return None, None
    if lo is None:
        return hi[0], 50
    if hi is None:
        return lo[0], 50
    if lo[1] == hi[1]:
        return lo[0], 50

    # Linear interpolation
    frac = (target_addr - lo[1]) / (hi[1] - lo[1])
    est_frame = lo[0] + frac * (hi[0] - lo[0])
    # Window proportional to frame span
    span = hi[0] - lo[0]
    window = max(int(span * 0.1), 30)  # at least 30 frames
    return int(est_frame), window


def compute_search_windows(missing_addrs, frame_addr_map):
    """Compute video frame search windows for each gap region.

    Groups contiguous missing addresses and computes a single window per group.
    Returns list of (start_vf, end_vf, [addr_list]).
    """
    if not missing_addrs:
        return []

    # Group contiguous addresses
    groups = []
    current_group = [missing_addrs[0]]
    for addr in missing_addrs[1:]:
        if addr - current_group[-1] <= 0x40:  # within 4 rows
            current_group.append(addr)
        else:
            groups.append(current_group)
            current_group = [addr]
    groups.append(current_group)

    print(f"  {len(groups)} gap groups from {len(missing_addrs)} missing addresses")

    windows = []
    for group in groups:
        first_addr = group[0]
        last_addr = group[-1]

        est_first, win_first = estimate_video_frame_for_address(first_addr, frame_addr_map)
        est_last, win_last = estimate_video_frame_for_address(last_addr, frame_addr_map)

        if est_first is None or est_last is None:
            continue

        start_vf = max(est_first - win_first, 0)
        end_vf = min(est_last + win_last, 93092)

        # Ensure minimum window size
        if end_vf - start_vf < 60:
            mid = (start_vf + end_vf) // 2
            start_vf = max(mid - 30, 0)
            end_vf = min(mid + 30, 93092)

        windows.append((start_vf, end_vf, group))

    # Merge overlapping windows
    windows.sort()
    merged = []
    for start, end, addrs in windows:
        if merged and start <= merged[-1][1] + 10:
            # Merge with previous
            prev_start, prev_end, prev_addrs = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_addrs + addrs)
        else:
            merged.append((start, end, addrs))

    total_frames = sum(e - s + 1 for s, e, _ in merged)
    print(f"  {len(merged)} search windows, {total_frames} total video frames to scan")

    return merged


def scan_video_windows(windows, missing_set, classifier):
    """Scan video frames in each window, collect observations for missing addresses.

    Returns dict: addr_int -> list of (video_frame, hex_bytes, byte_confs, addr_conf, row_y, gray_img)
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open full_video.mp4")
        sys.exit(1)

    observations = defaultdict(list)
    total_frames_read = 0
    total_unique = 0
    total_hits = 0

    for win_idx, (start_vf, end_vf, target_addrs) in enumerate(windows):
        target_set = set(target_addrs)
        window_size = end_vf - start_vf + 1

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_vf)

        prev_gray = None
        frames_read = 0
        unique_frames = 0
        hits = 0

        for vf in range(start_vf, end_vf + 1):
            ret, frame = cap.read()
            if not ret:
                break

            frames_read += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Skip duplicate frames
            if not is_frame_different(prev_gray, gray):
                continue

            unique_frames += 1
            prev_gray = gray

            # Process frame
            results = process_frame(classifier, gray)

            for addr_int, row_y, hex_bytes, byte_confs, addr_conf in results:
                if addr_int in missing_set:
                    observations[addr_int].append({
                        'video_frame': vf,
                        'hex_bytes': hex_bytes,
                        'byte_confs': byte_confs,
                        'addr_conf': addr_conf,
                        'row_y': row_y,
                        'gray': gray,
                    })
                    hits += 1

                # Also check C/D swap candidates for addresses with C or D
                addr_hex = f"{addr_int:05X}"
                if 'C' in addr_hex or 'D' in addr_hex:
                    from fix_d_c_misread import swap_candidates
                    for cand in swap_candidates(addr_hex):
                        cand_int = int(cand, 16)
                        if cand_int in missing_set and cand_int not in {r[0] for r in results}:  # r[0] = addr_int
                            # OCR read this as addr_int but it might really be cand_int
                            # Check if cand_int is closer to expected scroll position
                            observations[cand_int].append({
                                'video_frame': vf,
                                'hex_bytes': hex_bytes,
                                'byte_confs': byte_confs,
                                'addr_conf': addr_conf,
                                'gray': gray,
                                'ocr_addr': addr_int,  # original OCR read
                                'swapped': True,
                            })
                            hits += 1

        total_frames_read += frames_read
        total_unique += unique_frames
        total_hits += hits

        if hits > 0:
            recovered_in_window = len(set(obs['video_frame'] for a in target_set
                                         for obs in observations.get(a, [])))
            addrs_found = sum(1 for a in target_set if a in observations)
            print(f"  Window {win_idx+1}/{len(windows)}: vf {start_vf}-{end_vf} "
                  f"({unique_frames} unique), {addrs_found}/{len(target_addrs)} addrs found")

    cap.release()

    print(f"\n  Total: {total_frames_read} frames read, {total_unique} unique, "
          f"{total_hits} gap hits")
    print(f"  Addresses with observations: {len(observations)}")

    return observations


def find_row_y_for_address(gray, classifier, target_addr):
    """Find the row_y for a specific address in a frame.

    Returns row_y or None.
    """
    from template_matcher import CAL, ADDR_X_START
    from extract_pipeline import read_address_from_row

    y_first = CAL['first_row_center_y']
    row_h = CAL['row_height']
    num_rows = CAL['visible_rows']

    for row_idx in range(-2, num_rows):
        row_y = y_first + row_idx * row_h
        if row_y < 14 or row_y > gray.shape[0] - 14:
            continue
        addr_str, _ = read_address_from_row(classifier, gray, row_y, ADDR_X_START)
        try:
            addr_int = int(addr_str, 16)
            if addr_int == target_addr:
                return row_y
        except ValueError:
            pass
    return None


def save_crops_and_update_index(observations, crop_index, classifier):
    """Save row crops for recovered observations and update crop_index.json.

    Returns set of newly recovered addresses.
    """
    new_addrs = set()
    updated_addrs = set()

    for addr_int, obs_list in sorted(observations.items()):
        addr_hex = f"{addr_int:05X}"
        addr_lower = addr_hex.lower()

        # Filter to best observations (deduplicate by video frame)
        best_by_frame = {}
        for obs in obs_list:
            vf = obs['video_frame']
            if vf not in best_by_frame or obs['addr_conf'] > best_by_frame[vf]['addr_conf']:
                best_by_frame[vf] = obs

        if not best_by_frame:
            continue

        # Create crop directory
        crop_dir = os.path.join(CROPS_DIR, addr_lower)
        os.makedirs(crop_dir, exist_ok=True)

        # Initialize crop_index entry if needed
        is_new = addr_hex not in crop_index
        if is_new:
            crop_index[addr_hex] = {
                'frames': [],
                'video_frames': [],
                'readings': {},
                'confidences': {},
                'row_ys': {},
            }
            new_addrs.add(addr_hex)
        else:
            updated_addrs.add(addr_hex)

        entry = crop_index[addr_hex]
        entry.setdefault('video_frames', [])

        for vf, obs in sorted(best_by_frame.items()):
            gray = obs['gray']

            # Find row_y for crop extraction
            target = obs.get('ocr_addr', addr_int) if obs.get('swapped') else addr_int
            row_y = find_row_y_for_address(gray, classifier, target)
            if row_y is None:
                # Try the swapped address too
                if obs.get('swapped'):
                    row_y = find_row_y_for_address(gray, classifier, addr_int)
                if row_y is None:
                    continue

            # Save crop (asymmetric — shifted up to center on ink)
            row_y = int(row_y)
            y_top = max(0, row_y - 17)
            y_bot = min(gray.shape[0], row_y + 11)
            crop = gray[y_top:y_bot, 284:1120]

            crop_name = crop_filename(vf, is_video=True)
            crop_path = os.path.join(crop_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            # Update crop_index entry — video frames go in video_frames array
            if vf not in entry['video_frames']:
                entry['video_frames'].append(vf)

            vf_key = video_frame_key(vf)
            entry.setdefault('readings', {})[vf_key] = [
                b.upper() for b in obs['hex_bytes']
            ]
            entry.setdefault('confidences', {})[vf_key] = [
                round(float(c), 3) for c in obs['byte_confs']
            ]
            entry.setdefault('row_ys', {})[vf_key] = round(float(row_y), 1)

        # Sort frames
        entry['frames'] = sorted(entry['frames'])
        entry['video_frames'] = sorted(entry['video_frames'])

    print(f"  New addresses: {len(new_addrs)}")
    print(f"  Updated addresses: {len(updated_addrs)}")

    return new_addrs | updated_addrs


def weighted_majority_vote(readings, confidences):
    """Per-byte weighted majority voting across frames."""
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

    return result, len(readings)


def update_extracted_firmware(crop_index, affected_addrs):
    """Update extracted_firmware.txt with newly recovered addresses."""
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
    for addr_key in sorted(affected_addrs):
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

    with open(EXTRACTED_FW_PATH, 'w') as f:
        for h in header_lines:
            f.write(h + '\n')
        for addr_str in sorted(lines_by_addr.keys(),
                                key=lambda x: int(x, 16)):
            f.write(lines_by_addr[addr_str] + '\n')

    print(f"  Updated {updated} lines, added {added} lines in extracted_firmware.txt")


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


def reset_review_state(affected_addrs):
    """Reset review status for affected addresses."""
    if not os.path.exists(REVIEW_STATE_PATH):
        print("  review_state.json not found -- skipping")
        return

    with open(REVIEW_STATE_PATH) as f:
        state = json.load(f)

    if 'lines' not in state:
        return

    merged_bytes = {}
    if os.path.exists(MERGED_FW_PATH):
        with open(MERGED_FW_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split(':')
                if len(parts) >= 2:
                    addr_str = parts[0].strip().upper()
                    rest = re.sub(r'\[.*?\]', '', parts[1]).strip()
                    hex_bytes = rest.split()
                    if len(hex_bytes) == 16:
                        merged_bytes[addr_str] = hex_bytes

    reset_count = 0
    for addr_key in sorted(affected_addrs):
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


def main():
    print("=" * 60)
    print("Strategy 7: Full-Video Gap Recovery")
    print("=" * 60)

    # Step 1: Find missing addresses
    print("\nStep 1: Identify missing addresses")
    missing = find_missing_addresses()
    print(f"  {len(missing)} missing addresses in ROM range "
          f"(${ROM_MIN:05X}-${ROM_MAX:05X})")

    if not missing:
        print("  No gaps to recover!")
        return

    # Step 2: Build search windows from crop_index mapping
    print("\nStep 2: Build search windows")
    crop_index = load_crop_index()
    frame_addr_map = build_frame_address_map(crop_index)
    print(f"  {len(frame_addr_map)} frame-address points from crop_index")
    windows = compute_search_windows(missing, frame_addr_map)

    if not windows:
        print("  No search windows computed -- cannot proceed")
        return

    # Step 3: Load classifier and scan video
    print("\nStep 3: Scan video frames in search windows")
    classifier = FastKNNClassifier()
    classifier.load(CLASSIFIER_PATH)

    missing_set = set(missing)
    t0 = time.time()
    observations = scan_video_windows(windows, missing_set, classifier)
    t1 = time.time()
    print(f"  Scan completed in {t1 - t0:.1f}s")

    if not observations:
        print("\n  No gap addresses found in video. All gaps are likely unrecoverable")
        print("  (video jumps over those addresses)")
        return

    # Step 4: Save crops and update crop_index
    print("\nStep 4: Save crops and update crop_index.json")
    affected_addrs = save_crops_and_update_index(observations, crop_index, classifier)
    save_crop_index(crop_index)

    if not affected_addrs:
        print("  No addresses recovered")
        return

    # Log to frame_moves.json and frame_assignments.json
    recovered_map = {}
    for addr_hex in affected_addrs:
        entry = crop_index.get(addr_hex, {})
        recovered_map[addr_hex] = entry.get('video_frames', [])
    log_gap_recovery(recovered_map)
    update_frame_assignments(recovered_map, crop_index)

    # Step 5: Update extracted_firmware.txt
    print("\nStep 5: Update extracted_firmware.txt")
    update_extracted_firmware(crop_index, affected_addrs)

    # Step 6: Rebuild downstream
    print("\nStep 6: Rebuild downstream files")
    rebuild_downstream()

    # Step 7: Reset review state
    print("\nStep 7: Reset review state")
    reset_review_state(affected_addrs)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    recovered = len([a for a in missing if f"{a:05X}" in affected_addrs])
    still_missing = len(missing) - recovered
    print(f"  Addresses recovered: {recovered}/{len(missing)}")
    print(f"  Still missing: {still_missing}")

    if still_missing > 0:
        unrecovered = [a for a in missing if f"{a:05X}" not in affected_addrs]
        # Group contiguous unrecovered addresses
        groups = []
        current = [unrecovered[0]]
        for a in unrecovered[1:]:
            if a - current[-1] <= 0x10:
                current.append(a)
            else:
                groups.append(current)
                current = [a]
        groups.append(current)

        print(f"  Unrecoverable gap regions ({len(groups)} groups):")
        for g in groups[:20]:
            if len(g) == 1:
                print(f"    ${g[0]:05X}")
            else:
                print(f"    ${g[0]:05X}-${g[-1]:05X} ({len(g)} addrs)")
        if len(groups) > 20:
            print(f"    ... and {len(groups) - 20} more groups")

    print("=" * 60)


if __name__ == '__main__':
    main()
