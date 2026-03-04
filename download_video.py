#!/usr/bin/env python3
"""
Download the Xeltek SuperPro 6100N hex dump video and extract frames.

Video: https://www.youtube.com/watch?v=F3vQnaCocdQ
Segment: 13:41-24:50 (821s-1490s) — Xeltek Edit Buffer scrolling through R5F21258SNFP flash

Downloads the full video at max quality to preserve the original H.264 bitstream,
then extracts frames from the relevant segment without re-encoding.
"""

import argparse
import os
import subprocess
import sys


VIDEO_URL = "https://www.youtube.com/watch?v=F3vQnaCocdQ"
OUTPUT_FILE = "full_video.mp4"
FRAMES_DIR = "frames"
FPS = 30
SEGMENT_START = 821
SEGMENT_END = 1490


def download_full_video():
    """Download the full YouTube video at max quality (no re-encoding)."""
    if os.path.exists(OUTPUT_FILE):
        print(f"Video already exists: {OUTPUT_FILE}")
        return True

    cwd = os.getcwd()
    yt_dlp_path = os.path.join(cwd, "venv", "bin", "yt-dlp")

    if not os.path.exists(yt_dlp_path):
        yt_dlp_path = "yt-dlp"  # fall back to system

    command = [
        yt_dlp_path,
        "-f", "137+140",
        "-o", OUTPUT_FILE,
        VIDEO_URL,
    ]

    env = os.environ.copy()

    print("Downloading full video (format 137 H.264 1080p + format 140 AAC)...")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"Downloaded: {OUTPUT_FILE}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def extract_frames(video_path, output_dir=FRAMES_DIR, fps=FPS):
    """Extract frames from the original bitstream for the relevant segment."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if frames already exist
    existing = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if existing:
        print(f"Frames directory already has {len(existing)} frames, skipping extraction.")
        return True

    cwd = os.getcwd()
    local_ffmpeg = os.path.join(cwd, "ffmpeg")
    ffmpeg_cmd = local_ffmpeg if os.path.exists(local_ffmpeg) else "ffmpeg"

    # -ss after -i = decode-accurate seeking (no keyframe snapping)
    # Extracts directly from original bitstream — no re-encoding
    command = [
        ffmpeg_cmd,
        "-i", video_path,
        "-ss", str(SEGMENT_START),
        "-to", str(SEGMENT_END),
        "-vf", f"fps={fps}",
        os.path.join(output_dir, "frame_%05d.png"),
    ]

    print(f"Extracting frames at {fps} fps from {SEGMENT_START}s to {SEGMENT_END}s...")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        print(f"Extracted {frame_count} frames to {output_dir}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Frame extraction failed: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr[-500:]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Xeltek hex dump video and extract frames")
    parser.add_argument("--skip-download", action="store_true", help="Skip video download")
    parser.add_argument("--skip-extract", action="store_true", help="Skip frame extraction")
    parser.add_argument("--fps", type=int, default=FPS, help=f"Frame extraction rate (default: {FPS})")
    args = parser.parse_args()

    if not args.skip_download:
        if not download_full_video():
            sys.exit(1)

    if not args.skip_extract:
        video = OUTPUT_FILE
        if not os.path.exists(video):
            print(f"Error: video file not found: {video}")
            sys.exit(1)
        if not extract_frames(video, fps=args.fps):
            sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
