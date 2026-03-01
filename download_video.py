#!/usr/bin/env python3
"""
Download the Xeltek SuperPro 6100N hex dump video segment and extract frames.

Video: https://www.youtube.com/watch?v=F3vQnaCocdQ
Segment: 13:41-24:50 (821s-1490s) — Xeltek Edit Buffer scrolling through R5F21258SNFP flash
"""

import argparse
import os
import subprocess
import sys


VIDEO_URL = "https://www.youtube.com/watch?v=F3vQnaCocdQ"
OUTPUT_FILE = "xeltek_dump.mp4"
FRAMES_DIR = "frames"
FPS = 30


def download_video_segment():
    """Download the relevant video segment using yt-dlp."""
    if os.path.exists(OUTPUT_FILE):
        print(f"Video already exists: {OUTPUT_FILE}")
        return True

    cwd = os.getcwd()
    # Check for local ffmpeg binary first, then fall back to system PATH
    local_ffmpeg = os.path.join(cwd, "ffmpeg")
    yt_dlp_path = os.path.join(cwd, "venv", "bin", "yt-dlp")

    if not os.path.exists(yt_dlp_path):
        yt_dlp_path = "yt-dlp"  # fall back to system

    command = [
        yt_dlp_path,
        "--download-sections", "*821-1490",
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", OUTPUT_FILE,
        VIDEO_URL,
    ]

    if os.path.exists(local_ffmpeg):
        command.insert(1, "--ffmpeg-location")
        command.insert(2, local_ffmpeg)

    env = os.environ.copy()
    if os.path.exists(local_ffmpeg):
        env["PATH"] = f"{cwd}:{env.get('PATH', '')}"

    print(f"Downloading video segment (13:41-24:50)...")
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
    """Extract frames from video at the specified fps."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if frames already exist
    existing = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if existing:
        print(f"Frames directory already has {len(existing)} frames, skipping extraction.")
        return True

    cwd = os.getcwd()
    local_ffmpeg = os.path.join(cwd, "ffmpeg")
    ffmpeg_cmd = local_ffmpeg if os.path.exists(local_ffmpeg) else "ffmpeg"

    command = [
        ffmpeg_cmd,
        "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(output_dir, "frame_%05d.png"),
    ]

    print(f"Extracting frames at {fps} fps...")
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
        if not download_video_segment():
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
