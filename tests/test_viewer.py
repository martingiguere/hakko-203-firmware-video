"""Tests for the frame viewer route and JS logic specification."""

import math


# ---------------------------------------------------------------------------
# A. Flask Backend Tests
# ---------------------------------------------------------------------------

class TestViewerRoute:
    def test_viewer_returns_200(self, app_client):
        resp = app_client.get('/viewer')
        assert resp.status_code == 200

    def test_viewer_content_type_html(self, app_client):
        resp = app_client.get('/viewer')
        assert 'text/html' in resp.content_type

    def test_viewer_injects_segment_start(self, app_client):
        html = app_client.get('/viewer').data.decode()
        # The JS should contain the literal constant assignment
        assert 'const SEGMENT_START = 821' in html

    def test_viewer_injects_fps(self, app_client):
        html = app_client.get('/viewer').data.decode()
        assert 'const FPS = 30' in html

    def test_viewer_injects_total_frames(self, app_client):
        html = app_client.get('/viewer').data.decode()
        assert 'const TOTAL_FRAMES = 20070' in html

    def test_viewer_has_nav_buttons(self, app_client):
        html = app_client.get('/viewer').data.decode()
        for delta in ['-100', '-10', '-1', '1', '10', '100']:
            assert f'data-delta="{delta}"' in html

    def test_viewer_has_jump_inputs(self, app_client):
        html = app_client.get('/viewer').data.decode()
        assert 'id="jump-input"' in html
        assert 'id="jump-frame-input"' in html
        assert 'id="jump-btn"' in html
        assert 'id="jump-frame-btn"' in html

    def test_viewer_has_info_bar_elements(self, app_client):
        html = app_client.get('/viewer').data.decode()
        assert 'id="frame-num"' in html
        assert 'id="seg-time"' in html
        assert 'id="yt-time"' in html


class TestFrameApiRoute:
    def test_valid_frame_returns_png(self, app_client):
        resp = app_client.get('/api/frame/1')
        assert resp.status_code == 200
        assert resp.content_type == 'image/png'
        # Check PNG signature
        assert resp.data[:4] == b'\x89PNG'

    def test_missing_frame_returns_404(self, app_client):
        resp = app_client.get('/api/frame/99999')
        assert resp.status_code == 404
        assert resp.is_json
        assert 'error' in resp.get_json()

    def test_non_integer_frame_returns_404(self, app_client):
        resp = app_client.get('/api/frame/abc')
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# B. JavaScript Logic Specification Tests
# ---------------------------------------------------------------------------
# These validate the math that the JS IIFE implements, using the same formulas.

SEGMENT_START = 821
FPS = 30
TOTAL_FRAMES = 20070


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yt_seconds(frame):
    return SEGMENT_START + (frame - 1) / FPS


def fmt_time(total_sec):
    m = int(total_sec) // 60
    s = int(total_sec) % 60
    return f"{m}:{s:02d}"


def fmt_seg_time(total_sec):
    m = int(total_sec) // 60
    s = total_sec % 60
    # Match JS: toFixed(1) with leading zero pad
    s_str = f"{s:.1f}"
    if s < 10:
        s_str = "0" + s_str
    return f"{m}:{s_str}"


def timestamp_to_frame(yt_sec):
    offset_sec = yt_sec - SEGMENT_START
    return round(offset_sec * FPS) + 1


class TestYouTubeTimestamp:
    def test_frame_1_youtube_time(self):
        assert fmt_time(yt_seconds(1)) == "13:41"

    def test_frame_20070_youtube_time(self):
        # 821 + 20069/30 = 1489.966... -> 24:49
        assert fmt_time(yt_seconds(20070)) == "24:49"


class TestTimestampJump:
    def test_jump_16_40_to_frame(self):
        yt_sec = 16 * 60 + 40  # 1000s
        assert timestamp_to_frame(yt_sec) == 5371

    def test_jump_821s_to_frame_1(self):
        assert timestamp_to_frame(821) == 1

    def test_jump_segment_start_mm_ss(self):
        yt_sec = 13 * 60 + 41  # 821s
        assert timestamp_to_frame(yt_sec) == 1


class TestClamping:
    def test_clamp_below_min(self):
        assert clamp(0, 1, TOTAL_FRAMES) == 1

    def test_clamp_above_max(self):
        assert clamp(99999, 1, TOTAL_FRAMES) == TOTAL_FRAMES

    def test_clamp_in_range(self):
        assert clamp(500, 1, TOTAL_FRAMES) == 500


class TestSegmentTime:
    def test_frame_1_segment_time(self):
        offset = (1 - 1) / FPS  # 0.0
        assert fmt_seg_time(offset) == "0:00.0"

    def test_frame_20070_segment_time(self):
        offset = (20070 - 1) / FPS  # 668.966...
        result = fmt_seg_time(offset)
        # 668.966s = 11m 8.966s -> "11:09.0" (toFixed(1) rounds 8.966 to 9.0)
        assert result == "11:09.0"

    def test_frame_901_segment_time(self):
        # (901-1)/30 = 30.0s -> "0:30.0"
        offset = (901 - 1) / FPS
        assert fmt_seg_time(offset) == "0:30.0"
