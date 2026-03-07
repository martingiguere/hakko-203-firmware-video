#!/usr/bin/env python3
"""
Flask backend for the Firmware Review Tool — Hakko FM-203.

Serves frame crop images, provides line data / confidence / OCR readings via JSON API,
and handles save/export operations.
"""

import os
import json
import datetime
import argparse
import shutil

from flask import Flask, jsonify, request, send_file, render_template

from crc import compute_byte_sum_32, build_firmware_binary, build_rom_binary, \
    BASE_ADDR, BUFFER_SIZE, EXPECTED_CHECKSUM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CROPS_DIR = os.path.join(PROJECT_ROOT, 'crops')
CROP_INDEX_PATH = os.path.join(CROPS_DIR, 'crop_index.json')
GAP_INDEX_PATH = os.path.join(CROPS_DIR, 'gap_context_index.json')
GAP_CROPS_DIR = os.path.join(CROPS_DIR, 'gap')
REF_CROPS_DIR = os.path.join(CROPS_DIR, 'ref')
MERGED_PATH = os.path.join(PROJECT_ROOT, 'firmware_merged.txt')
FRAMES_DIR = os.path.join(PROJECT_ROOT, 'frames')
REVIEW_STATE_PATH = os.path.join(PROJECT_ROOT, 'review_state.json')
EXPORT_HEX_PATH = os.path.join(PROJECT_ROOT, 'firmware_reviewed.txt')
EXPORT_ROM_PATH = os.path.join(PROJECT_ROOT, 'hakko_fm203_reviewed.bin')
EXPORT_FULL_PATH = os.path.join(PROJECT_ROOT, 'hakko_fm203_full_reviewed.bin')
FRAME_MOVES_PATH = os.path.join(PROJECT_ROOT, 'frame_moves.json')

# Confirmed erased flash regions — all-FF, no review needed.
FF_FORCED_REGIONS = [
    (0x00000, 0x03FFF),
]

END_ADDR = 0x13FF0
ALL_ADDRESSES = [f"{a:05X}" for a in range(0, 0x14000, 0x10)]
TOTAL_LINES = len(ALL_ADDRESSES)  # 5120

CHECKSUM_TARGET = 0x00D2F2FF

# ISP ID code bytes at known addresses
ISP_ID_BYTES = {
    0x0FFE3: 0x4B,
    0x0FFEB: 0x30,
    0x0FFEF: 0x30,
    0x0FFF3: 0x32,
    0x0FFF7: 0x35,
    0x0FFFB: 0x36,
}

app = Flask(__name__)

# --- In-memory state ---
crop_index = {}       # addr_upper -> {frames: [...], readings: {frame: [bytes]}}
gap_context_index = {}  # addr_upper -> {gap_id, gap_start, gap_end, gap_size, candidates: [...]}
review_state = {}     # Full review state dict (lines, settings, stats, etc.)
minimap_cache = None  # Cached minimap data, invalidated on status change
dirty = False         # Unsaved changes flag
ref_addresses = set()
frame_moves = []      # In-memory list of frame move records


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def load_crop_index():
    global crop_index, ref_addresses
    if os.path.exists(CROP_INDEX_PATH):
        with open(CROP_INDEX_PATH, encoding='utf-8') as f:
            crop_index = json.load(f)
        ref_list = crop_index.pop("ref_addresses", [])
        ref_addresses = set(a.upper() for a in ref_list)
        if not ref_addresses and os.path.isdir(REF_CROPS_DIR):
            for fname in os.listdir(REF_CROPS_DIR):
                if fname.endswith('.png'):
                    ref_addresses.add(fname[:-4].upper())
        print(f"Loaded crop index: {len(crop_index)} addresses, {len(ref_addresses)} ref addresses")
    else:
        print("WARNING: crop_index.json not found. Run precompute.py first.")
        crop_index = {}


def load_gap_context_index():
    global gap_context_index
    if os.path.exists(GAP_INDEX_PATH):
        with open(GAP_INDEX_PATH, encoding='utf-8') as f:
            gap_context_index = json.load(f)
        print(f"Loaded gap context index: {len(gap_context_index)} addresses")
    else:
        print("WARNING: gap_context_index.json not found. Run precompute_gaps.py first.")
        gap_context_index = {}


def load_frame_moves():
    global frame_moves
    if os.path.exists(FRAME_MOVES_PATH):
        with open(FRAME_MOVES_PATH, encoding='utf-8') as f:
            data = json.load(f)
        frame_moves = data.get("moves", [])
        print(f"Loaded frame moves ledger: {len(frame_moves)} moves")
    else:
        frame_moves = []


def save_frame_moves():
    with open(FRAME_MOVES_PATH, 'w', encoding='utf-8') as f:
        json.dump({"moves": frame_moves}, f, indent=2)


def save_crop_index():
    data = dict(crop_index)
    data["ref_addresses"] = sorted(ref_addresses)
    tmp_path = CROP_INDEX_PATH + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    os.replace(tmp_path, CROP_INDEX_PATH)


def weighted_majority_vote(readings, confidences):
    """Weighted majority vote across frames for 16 byte positions."""
    if not readings:
        return None
    result = []
    for byte_idx in range(16):
        votes = {}
        for frame_str, byte_list in readings.items():
            if byte_idx < len(byte_list):
                val = byte_list[byte_idx]
                if val == "--":
                    continue
                weight = 1.0
                if frame_str in confidences:
                    conf_list = confidences[frame_str]
                    if byte_idx < len(conf_list):
                        weight = conf_list[byte_idx]
                votes[val] = votes.get(val, 0) + weight
        if votes:
            result.append(max(votes, key=votes.get))
        else:
            result.append("--")
    return result


def apply_single_move(frame, from_addr, to_addr):
    """Move a frame's data within the in-memory crop_index."""
    frame_str = str(frame)
    src = crop_index.get(from_addr)
    if not src:
        return False

    # Create destination entry if needed
    if to_addr not in crop_index:
        crop_index[to_addr] = {"frames": [], "readings": {}, "confidences": {}}
    dst = crop_index[to_addr]

    # Move readings and confidences
    if frame_str in src.get("readings", {}):
        dst.setdefault("readings", {})[frame_str] = src["readings"].pop(frame_str)
    if frame_str in src.get("confidences", {}):
        dst.setdefault("confidences", {})[frame_str] = src["confidences"].pop(frame_str)

    # Move frame number
    if frame in src.get("frames", []):
        src["frames"].remove(frame)
    if frame not in dst["frames"]:
        dst["frames"].append(frame)
        dst["frames"].sort()

    # Move crop PNG
    src_png = os.path.join(CROPS_DIR, from_addr.lower(), f"frame_{frame:05d}.png")
    dst_dir = os.path.join(CROPS_DIR, to_addr.lower())
    dst_png = os.path.join(dst_dir, f"frame_{frame:05d}.png")
    if os.path.exists(src_png):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src_png, dst_png)

    # Clean up empty source entry/directory
    if not src["frames"] and not src.get("readings"):
        del crop_index[from_addr]
        src_dir = os.path.join(CROPS_DIR, from_addr.lower())
        if os.path.isdir(src_dir) and not os.listdir(src_dir):
            os.rmdir(src_dir)

    return True


def recompute_consensus_for_addr(addr):
    """Recompute weighted majority vote for an address and update review_state."""
    entry = crop_index.get(addr)
    if not entry or not entry.get("readings"):
        return
    consensus = weighted_majority_vote(entry["readings"], entry.get("confidences", {}))
    if consensus and addr in review_state.get("lines", {}):
        line = review_state["lines"][addr]
        # Only overwrite if user hasn't manually edited
        if line.get("source") != "user":
            line["bytes"] = consensus
            line["source"] = "merged"
            line["status"] = "unreviewed"
            line["edited_positions"] = []


def parse_merged_file():
    """Parse firmware_merged.txt into {addr_upper: [byte_hex, ...]}."""
    lines = {}
    with open(MERGED_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if ':' not in line:
                continue
            parts = line.split(':')
            addr_str = parts[0].strip().upper().zfill(5)
            hex_part = parts[1].strip()
            # Strip trailing [REF] or other annotations
            if '[' in hex_part:
                hex_part = hex_part[:hex_part.index('[')].strip()
            byte_list = hex_part.split()
            if len(byte_list) >= 16:
                lines[addr_str] = [b.upper() for b in byte_list[:16]]
    return lines


def init_state_from_merged():
    """Initialize review state from firmware_merged.txt."""
    global review_state
    merged = parse_merged_file()
    lines = {}
    for addr in ALL_ADDRESSES:
        if addr in merged:
            lines[addr] = {
                "status": "unreviewed",
                "bytes": merged[addr],
                "source": "merged",
                "edited_positions": [],
            }
        else:
            lines[addr] = {
                "status": "unreviewed",
                "bytes": ["--"] * 16,
                "source": "missing",
                "edited_positions": [],
            }
    review_state = {
        "version": 1,
        "last_saved": None,
        "settings": {
            "zoom_level": 2,
            "current_address": "00000",
            "sort_order": "priority",
        },
        "lines": lines,
    }
    print(f"Initialized state from merged file: {len(merged)} data lines, "
          f"{TOTAL_LINES - len(merged)} missing")


def load_review_state():
    """Load review state from JSON or initialize from merged file."""
    global review_state
    if os.path.exists(REVIEW_STATE_PATH):
        with open(REVIEW_STATE_PATH, encoding='utf-8') as f:
            review_state = json.load(f)
        print(f"Loaded review state: {len(review_state.get('lines', {}))} lines")
        # Ensure all addresses exist (in case new addresses were added)
        for addr in ALL_ADDRESSES:
            if addr not in review_state["lines"]:
                review_state["lines"][addr] = {
                    "status": "unreviewed",
                    "bytes": ["--"] * 16,
                    "source": "missing",
                    "edited_positions": [],
                }
    else:
        init_state_from_merged()


def normalize_addr(addr):
    """Normalize address to uppercase 5-digit hex."""
    return addr.upper().zfill(5)


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def compute_byte_confidence(addr):
    """
    Compute per-byte confidence from cross-frame kNN agreement and kNN confidence.

    Single-frame addresses use kNN confidence directly (no cross-frame agreement).
    Multi-frame addresses use agreement-based approach, excluding "--" readings,
    with kNN confidence as a secondary signal.

    Returns list of 16 confidence levels: "high", "medium", "low", or "none".
    """
    entry = crop_index.get(addr)
    if not entry or not entry.get("readings"):
        return ["none"] * 16

    readings = entry["readings"]  # {frame_str: [16 bytes]}
    knn_confs = entry.get("confidences", {})  # {frame_str: [16 floats]}
    n_frames = len(readings)
    if n_frames == 0:
        return ["none"] * 16

    confidences = []

    if n_frames == 1:
        # Single-frame: use kNN confidence directly
        frame_str = next(iter(readings))
        frame_readings = readings[frame_str]
        frame_confs = knn_confs.get(frame_str, [0.0] * 16)
        for byte_idx in range(16):
            if byte_idx < len(frame_readings) and frame_readings[byte_idx] == "--":
                confidences.append("none")
            elif byte_idx < len(frame_confs):
                c = frame_confs[byte_idx]
                if c >= 0.7:
                    confidences.append("high")
                elif c >= 0.4:
                    confidences.append("medium")
                else:
                    confidences.append("low")
            else:
                confidences.append("none")
    else:
        # Multi-frame: agreement-based with kNN secondary signal
        for byte_idx in range(16):
            votes = {}
            knn_vals = []
            for frame_str, frame_readings in readings.items():
                if byte_idx < len(frame_readings):
                    val = frame_readings[byte_idx]
                    if val == "--":
                        continue  # Exclude blank readings
                    votes[val] = votes.get(val, 0) + 1
                    frame_conf = knn_confs.get(frame_str, [0.0] * 16)
                    if byte_idx < len(frame_conf):
                        knn_vals.append(frame_conf[byte_idx])

            if not votes:
                confidences.append("none")
                continue

            total_valid = sum(votes.values())
            best_count = max(votes.values())
            agreement = best_count / total_valid
            avg_knn = sum(knn_vals) / len(knn_vals) if knn_vals else 0.0

            if agreement >= 0.9 and avg_knn >= 0.3:
                confidences.append("high")
            elif agreement >= 0.6:
                confidences.append("medium")
            else:
                confidences.append("low")

    return confidences


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats():
    """Compute review statistics."""
    lines = review_state["lines"]
    stats = {
        "total_lines": TOTAL_LINES,
        "missing": 0,
        "unreviewed": 0,
        "accepted": 0,
        "edited": 0,
        "flagged": 0,
        "has_data": 0,
    }

    # ISP ID code check
    id_match = 0
    id_total = len(ISP_ID_BYTES)

    for addr, line in lines.items():
        status = line["status"]
        source = line.get("source", "")
        if status == "accepted":
            stats["accepted"] += 1
        elif status == "edited":
            stats["edited"] += 1
        elif status == "flagged":
            stats["flagged"] += 1
        else:
            stats["unreviewed"] += 1

        if source == "missing" and status == "unreviewed":
            stats["missing"] += 1
        if any(b != "--" for b in line["bytes"]):
            stats["has_data"] += 1

    # Check ISP ID bytes against current firmware state
    for byte_addr, expected_val in ISP_ID_BYTES.items():
        line_addr = (byte_addr // 0x10) * 0x10
        byte_offset = byte_addr % 0x10
        addr_key = f"{line_addr:05X}"
        line = lines.get(addr_key, {})
        byte_list = line.get("bytes", ["--"] * 16)
        if byte_offset < len(byte_list) and byte_list[byte_offset] != "--":
            try:
                if int(byte_list[byte_offset], 16) == expected_val:
                    id_match += 1
            except ValueError:
                pass

    stats["id_match"] = id_match
    stats["id_total"] = id_total

    return stats


# ---------------------------------------------------------------------------
# Minimap
# ---------------------------------------------------------------------------

def build_minimap():
    """Build minimap data: list of 5,120 entries with address, status tier, and confidence."""
    lines = review_state["lines"]
    minimap = []
    for addr in ALL_ADDRESSES:
        line = lines.get(addr, {})
        status = line.get("status", "unreviewed")
        source = line.get("source", "missing")

        # Determine tier for minimap coloring
        if status == "accepted":
            tier = "accepted"
        elif status == "edited":
            tier = "edited"
        elif status == "flagged":
            tier = "flagged"
        elif source == "missing":
            tier = "missing"
        else:
            # Unreviewed with data -- check confidence
            addr_int = int(addr, 16)
            in_ff_region = any(s <= addr_int <= e for s, e in FF_FORCED_REGIONS)
            if in_ff_region:
                tier = "high_confidence"
            else:
                confs = compute_byte_confidence(addr)
                low_count = sum(1 for c in confs if c in ("low", "none"))
                if low_count >= 4:
                    tier = "low_confidence"
                else:
                    tier = "high_confidence"

        entry = {"address": addr, "tier": tier}
        if tier == "missing":
            gap_entry = gap_context_index.get(addr)
            if gap_entry:
                entry["gap_start"] = gap_entry["gap_start"]
        minimap.append(entry)
    return minimap


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/viewer')
def viewer():
    return render_template('viewer.html',
                           SEGMENT_START=821,
                           FPS=30,
                           TOTAL_FRAMES=20070)


@app.route('/api/line/<addr>')
def get_line(addr):
    addr = normalize_addr(addr)
    line = review_state["lines"].get(addr)
    if not line:
        return jsonify({"error": "Address not found"}), 404

    confidence = compute_byte_confidence(addr)
    entry = crop_index.get(addr, {})
    frames = entry.get("frames", [])

    return jsonify({
        "address": addr,
        "bytes": line["bytes"],
        "status": line["status"],
        "source": line.get("source", "unknown"),
        "edited_positions": line.get("edited_positions", []),
        "confidence": confidence,
        "frame_count": len(frames),
        "frames": frames,
    })


@app.route('/api/frames/<addr>')
def get_frames(addr):
    addr = normalize_addr(addr)
    entry = crop_index.get(addr, {})
    frames = entry.get("frames", [])
    readings = entry.get("readings", {})
    knn_confidences = entry.get("confidences", {})
    return jsonify({
        "address": addr,
        "frames": frames,
        "readings": readings,
        "knn_confidences": knn_confidences,
        "has_ref": addr in ref_addresses,
    })


@app.route('/api/crop/<addr>/<int:frame>')
def get_crop(addr, frame):
    addr_lower = normalize_addr(addr).lower()
    crop_path = os.path.join(CROPS_DIR, addr_lower, f'frame_{frame:05d}.png')
    if not os.path.exists(crop_path):
        return jsonify({"error": "Crop not found"}), 404
    return send_file(crop_path, mimetype='image/png')


@app.route('/api/ref_crop/<addr>')
def get_ref_crop(addr):
    addr = normalize_addr(addr)
    if addr not in ref_addresses:
        return jsonify({"error": "No reference crop"}), 404
    crop_path = os.path.join(REF_CROPS_DIR, f'{addr.lower()}.png')
    if not os.path.exists(crop_path):
        return jsonify({"error": "Reference crop file not found"}), 404
    return send_file(crop_path, mimetype='image/png')


@app.route('/api/frame/<int:frame>')
def get_frame(frame):
    frame_path = os.path.join(FRAMES_DIR, f'frame_{frame:05d}.png')
    if not os.path.exists(frame_path):
        return jsonify({"error": "Frame not found"}), 404
    return send_file(frame_path, mimetype='image/png')


@app.route('/api/minimap')
def get_minimap():
    global minimap_cache
    if minimap_cache is None:
        minimap_cache = build_minimap()
    return jsonify(minimap_cache)


@app.route('/api/stats')
def get_stats():
    stats = compute_stats()
    # Compute live checksum
    lines_dict = {}
    for addr, line in review_state["lines"].items():
        if any(b != "--" for b in line["bytes"]):
            lines_dict[addr] = line["bytes"]
    firmware = build_firmware_binary(lines_dict)
    checksum = compute_byte_sum_32(firmware)
    stats["checksum"] = f"{checksum:08X}"
    stats["checksum_target"] = f"{CHECKSUM_TARGET:08X}"
    stats["checksum_match"] = checksum == CHECKSUM_TARGET
    stats["dirty"] = dirty
    return jsonify(stats)


@app.route('/api/line/<addr>', methods=['POST'])
def update_line(addr):
    global dirty, minimap_cache
    addr = normalize_addr(addr)
    line = review_state["lines"].get(addr)
    if not line:
        return jsonify({"error": "Address not found"}), 404

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    old_status = line["status"]

    # Update bytes if provided
    if "bytes" in data:
        new_bytes = data["bytes"]
        if len(new_bytes) != 16:
            return jsonify({"error": "Must provide exactly 16 bytes"}), 400
        # Track which positions were edited
        edited = list(line.get("edited_positions", []))
        for i, (old_b, new_b) in enumerate(zip(line["bytes"], new_bytes)):
            if old_b != new_b and i not in edited:
                edited.append(i)
        line["bytes"] = [b.upper() for b in new_bytes]
        line["edited_positions"] = sorted(edited)
        if edited:
            line["source"] = "user"
            if line["status"] not in ("accepted", "flagged"):
                line["status"] = "edited"

    # Update status if provided
    if "status" in data:
        line["status"] = data["status"]

    dirty = True
    if line["status"] != old_status:
        minimap_cache = None  # Invalidate

    return jsonify({"ok": True, "address": addr, "status": line["status"]})


@app.route('/api/save', methods=['POST'])
def save_state():
    global dirty
    review_state["last_saved"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    review_state["stats"] = compute_stats()
    with open(REVIEW_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(review_state, f)
    dirty = False
    size_kb = os.path.getsize(REVIEW_STATE_PATH) / 1024
    return jsonify({"ok": True, "size_kb": round(size_kb, 1)})


@app.route('/api/export/hex', methods=['POST'])
def export_hex():
    lines = review_state["lines"]
    count = 0
    with open(EXPORT_HEX_PATH, 'w', encoding='utf-8') as f:
        f.write("# Hakko FM-203 Firmware - Reviewed extraction\n")
        f.write(f"# Exported: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        f.write("#\n")
        for addr in ALL_ADDRESSES:
            line = lines.get(addr, {})
            byte_list = line.get("bytes", ["--"] * 16)
            if any(b != "--" for b in byte_list):
                hex_str = ' '.join(byte_list)
                f.write(f"{addr}: {hex_str}\n")
                count += 1
    return jsonify({"ok": True, "path": EXPORT_HEX_PATH, "lines": count})


@app.route('/api/export/binary', methods=['POST'])
def export_binary():
    lines_dict = {}
    for addr, line in review_state["lines"].items():
        if any(b != "--" for b in line["bytes"]):
            lines_dict[addr] = line["bytes"]

    # ROM-only binary (64 KB: $04000-$13FFF)
    rom = build_rom_binary(lines_dict)
    with open(EXPORT_ROM_PATH, 'wb') as f:
        f.write(rom)

    # Full buffer binary (80 KB: $00000-$13FFF)
    full = build_firmware_binary(lines_dict)
    with open(EXPORT_FULL_PATH, 'wb') as f:
        f.write(full)

    checksum = compute_byte_sum_32(full)
    return jsonify({
        "ok": True,
        "rom_path": EXPORT_ROM_PATH,
        "full_path": EXPORT_FULL_PATH,
        "rom_size": len(rom),
        "full_size": len(full),
        "checksum": f"{checksum:08X}",
        "checksum_match": checksum == CHECKSUM_TARGET,
    })


@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(review_state.get("settings", {}))


@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    settings = review_state.setdefault("settings", {})
    for key in ("zoom_level", "current_address", "sort_order"):
        if key in data:
            settings[key] = data[key]
    return jsonify({"ok": True, "settings": settings})


# ---------------------------------------------------------------------------
# Gap Context Routes
# ---------------------------------------------------------------------------

@app.route('/api/gap_frames/<addr>')
def get_gap_frames(addr):
    addr = normalize_addr(addr)
    entry = gap_context_index.get(addr)
    if not entry:
        return jsonify({"error": "No gap data for this address"}), 404

    # Only include candidates where crop was saved
    saved_candidates = [c for c in entry.get("candidates", []) if c.get("crop_saved")]

    return jsonify({
        "address": addr,
        "gap_id": entry["gap_id"],
        "gap_start": entry["gap_start"],
        "gap_end": entry["gap_end"],
        "gap_size": entry["gap_size"],
        "candidates": saved_candidates,
    })


@app.route('/api/gap_context/<addr>/<int:frame>')
def get_gap_context(addr, frame):
    addr_lower = normalize_addr(addr).lower()
    crop_path = os.path.join(GAP_CROPS_DIR, addr_lower, f'frame_{frame:05d}.png')
    if not os.path.exists(crop_path):
        return jsonify({"error": "Gap context crop not found"}), 404
    return send_file(crop_path, mimetype='image/png')


@app.route('/api/gaps')
def get_gaps():
    # Group addresses by gap_id
    gaps_by_id = {}
    for addr, entry in gap_context_index.items():
        gid = entry["gap_id"]
        if gid not in gaps_by_id:
            gaps_by_id[gid] = {
                "gap_id": gid,
                "start": entry["gap_start"],
                "end": entry["gap_end"],
                "size": entry["gap_size"],
                "addresses": [],
            }
        line_state = review_state.get("lines", {}).get(addr, {})
        gaps_by_id[gid]["addresses"].append({
            "addr": addr,
            "status": line_state.get("status", "unreviewed"),
        })

    # Sort gaps by gap_id; sort addresses within each gap
    result = []
    for gid in sorted(gaps_by_id.keys()):
        gap = gaps_by_id[gid]
        gap["addresses"].sort(key=lambda x: x["addr"])
        result.append(gap)

    return jsonify({"gaps": result})


# ---------------------------------------------------------------------------
# Frame Move Routes
# ---------------------------------------------------------------------------

@app.route('/api/move_frame', methods=['POST'])
def move_frame():
    global dirty, minimap_cache
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    frame = data.get("frame")
    from_addr = normalize_addr(str(data.get("from_addr", "")))
    to_addr = normalize_addr(str(data.get("to_addr", "")))

    if frame is None:
        return jsonify({"error": "Missing 'frame'"}), 400

    # Validate source
    src = crop_index.get(from_addr)
    if not src or str(frame) not in src.get("readings", {}):
        return jsonify({"error": f"Frame {frame} not found in {from_addr}"}), 404

    # Validate destination address
    try:
        to_int = int(to_addr, 16)
    except ValueError:
        return jsonify({"error": f"Invalid address: {to_addr}"}), 400
    if to_int % 0x10 != 0 or to_int < 0 or to_int > END_ADDR:
        return jsonify({"error": f"Address {to_addr} not aligned to 0x10 or out of range"}), 400

    if from_addr == to_addr:
        return jsonify({"error": "Source and destination are the same"}), 400

    apply_single_move(frame, from_addr, to_addr)

    frame_moves.append({
        "frame": frame,
        "from_addr": from_addr,
        "to_addr": to_addr,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    save_frame_moves()
    save_crop_index()

    recompute_consensus_for_addr(from_addr)
    recompute_consensus_for_addr(to_addr)

    dirty = True
    minimap_cache = None

    return jsonify({"ok": True, "frame": frame, "from_addr": from_addr, "to_addr": to_addr})


@app.route('/api/move_frames', methods=['POST'])
def move_frames_batch():
    global dirty, minimap_cache
    data = request.get_json(silent=True)
    if not data or "moves" not in data:
        return jsonify({"error": "No 'moves' array in body"}), 400

    moves = data["moves"]
    affected_addrs = set()
    applied = 0

    for move in moves:
        frame = move.get("frame")
        from_addr = normalize_addr(str(move.get("from_addr", "")))
        to_addr = normalize_addr(str(move.get("to_addr", "")))

        if frame is None or from_addr == to_addr:
            continue

        src = crop_index.get(from_addr)
        if not src or str(frame) not in src.get("readings", {}):
            continue

        try:
            to_int = int(to_addr, 16)
        except ValueError:
            continue
        if to_int % 0x10 != 0 or to_int < 0 or to_int > END_ADDR:
            continue

        apply_single_move(frame, from_addr, to_addr)
        frame_moves.append({
            "frame": frame,
            "from_addr": from_addr,
            "to_addr": to_addr,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        affected_addrs.add(from_addr)
        affected_addrs.add(to_addr)
        applied += 1

    if applied:
        save_frame_moves()
        save_crop_index()
        for addr in affected_addrs:
            recompute_consensus_for_addr(addr)
        dirty = True
        minimap_cache = None

    return jsonify({"ok": True, "applied": applied, "total": len(moves)})


@app.route('/api/suggest_address/<int:frame>')
def suggest_address(frame):
    """Suggest the best address for a frame based on byte-match against consensus."""
    frame_str = str(frame)

    # Find all addresses that contain this frame
    candidates = []
    for addr, entry in crop_index.items():
        if frame_str in entry.get("readings", {}):
            candidates.append(addr)

    if not candidates:
        return jsonify({"error": f"Frame {frame} not found in any address"}), 404

    best_addr = None
    best_score = -1

    for addr in candidates:
        entry = crop_index[addr]
        reading = entry["readings"][frame_str]

        # Compute consensus excluding this frame
        other_readings = {k: v for k, v in entry["readings"].items() if k != frame_str}
        other_confs = {k: v for k, v in entry.get("confidences", {}).items() if k != frame_str}
        if not other_readings:
            continue

        consensus = weighted_majority_vote(other_readings, other_confs)
        if not consensus:
            continue

        # Count matching bytes
        score = sum(1 for i in range(16) if i < len(reading) and reading[i] == consensus[i]
                    and reading[i] != "--")
        if score > best_score:
            best_score = score
            best_addr = addr

    if best_addr:
        return jsonify({"suggested_addr": best_addr, "match_score": best_score, "candidates": candidates})
    return jsonify({"suggested_addr": None, "candidates": candidates})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8087


def main():
    parser = argparse.ArgumentParser(description='Firmware Review Tool — Hakko FM-203')
    parser.add_argument('-p', '--port', type=int, default=DEFAULT_PORT,
                        help=f'Port to listen on (default: {DEFAULT_PORT})')
    args = parser.parse_args()

    print("=== Firmware Review Tool — Hakko FM-203 ===")
    print()
    load_crop_index()
    load_frame_moves()
    load_gap_context_index()
    load_review_state()
    stats = compute_stats()
    print(f"\nStats: {stats['has_data']}/{TOTAL_LINES} lines with data, "
          f"{stats['missing']} missing, {stats['accepted']} accepted, "
          f"{stats['edited']} edited, {stats['flagged']} flagged")
    print(f"ID code: {stats['id_match']}/{stats['id_total']} bytes match")
    print(f"\nStarting server on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
