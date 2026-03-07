#!/usr/bin/env python3
"""
Utilities for distinguishing extracted vs full-video frame numbers.

Extracted frames (1-20,070): from frames/ PNGs, written by precompute.py
Full video frames (1-93,093): from full_video.mp4, written by fullvideo_gap_recovery.py

In crop_index.json:
  - "frames" array: extracted frame integers only
  - "video_frames" array: full-video frame integers
  - Dict keys in readings/confidences: bare string for extracted ("1234"),
    "v"-prefixed for video ("v35040")
  - Crop PNGs: frame_01234.png for extracted, frame_v35040.png for video
"""

import os
import re

MAX_EXTRACTED_FRAME = 20070


def is_video_frame_key(key):
    """Check if a readings/confidences dict key refers to a video frame."""
    return key.startswith('v')


def video_frame_key(vf):
    """Convert a video frame integer to a dict key string."""
    return f"v{vf}"


def extracted_frame_key(ef):
    """Convert an extracted frame integer to a dict key string."""
    return str(ef)


def crop_filename(frame_num, is_video=False):
    """Return the crop PNG filename for a frame."""
    if is_video:
        return f"frame_v{frame_num:05d}.png"
    return f"frame_{frame_num:05d}.png"


def frame_key(frame_num, is_video=False):
    """Return the dict key for a frame number."""
    if is_video:
        return video_frame_key(frame_num)
    return extracted_frame_key(frame_num)


def parse_frame_key(key):
    """Parse a dict key into (frame_int, is_video)."""
    if key.startswith('v'):
        return int(key[1:]), True
    return int(key), False


def all_frame_keys(entry):
    """Return all dict keys (extracted + video) for an entry's readings."""
    keys = []
    for ef in entry.get('frames', []):
        keys.append(extracted_frame_key(ef))
    for vf in entry.get('video_frames', []):
        keys.append(video_frame_key(vf))
    return keys


def total_frame_count(entry):
    """Total number of frames (extracted + video) for an entry."""
    return len(entry.get('frames', [])) + len(entry.get('video_frames', []))


def migrate_crop_index(crop_index, crops_dir):
    """One-time migration: separate video frames from extracted frames.

    Any frame > MAX_EXTRACTED_FRAME in 'frames' is moved to 'video_frames',
    and its readings/confidences keys are re-prefixed with 'v'.
    Crop PNGs are renamed from frame_NNNNN.png to frame_vNNNNN.png.

    Returns number of frames migrated.
    """
    migrated = 0
    for addr, entry in crop_index.items():
        if addr == 'ref_addresses':
            continue
        if not isinstance(entry, dict):
            continue

        frames = entry.get('frames', [])
        video_frames = entry.get('video_frames', [])

        extracted = []
        new_video = []
        for f in frames:
            if f > MAX_EXTRACTED_FRAME:
                new_video.append(f)
            else:
                extracted.append(f)

        if not new_video:
            continue

        entry['frames'] = sorted(extracted)
        entry.setdefault('video_frames', [])
        for vf in new_video:
            if vf not in entry['video_frames']:
                entry['video_frames'].append(vf)
        entry['video_frames'] = sorted(entry['video_frames'])

        # Re-key readings and confidences
        readings = entry.get('readings', {})
        confidences = entry.get('confidences', {})

        for vf in new_video:
            old_key = str(vf)
            new_key = video_frame_key(vf)

            if old_key in readings:
                readings[new_key] = readings.pop(old_key)
            if old_key in confidences:
                confidences[new_key] = confidences.pop(old_key)

            # Rename crop PNG
            addr_lower = addr.lower()
            crop_dir = os.path.join(crops_dir, addr_lower)
            old_png = os.path.join(crop_dir, f"frame_{vf:05d}.png")
            new_png = os.path.join(crop_dir, crop_filename(vf, is_video=True))
            if os.path.exists(old_png):
                os.rename(old_png, new_png)

            migrated += 1

    return migrated


def migrate_frame_moves(moves_path):
    """Migrate frame_moves.json: add 'v' prefix to video frame entries.

    Returns number of moves migrated.
    """
    import json

    if not os.path.exists(moves_path):
        return 0

    with open(moves_path) as f:
        data = json.load(f)

    moves = data.get('moves', [])
    migrated = 0
    for move in moves:
        frame = move.get('frame')
        if isinstance(frame, int) and frame > MAX_EXTRACTED_FRAME:
            move['frame'] = video_frame_key(frame)
            migrated += 1

    if migrated:
        with open(moves_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    return migrated
