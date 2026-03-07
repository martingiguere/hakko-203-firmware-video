"""Shared fixtures for viewer tests."""

import os
import sys
import struct
import tempfile
import pytest

# Add project root and firmware_review_tool/ to sys.path so imports work
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'firmware_review_tool'))


def _make_minimal_png(path):
    """Create a minimal valid 1x1 white PNG file."""
    # Minimal PNG: signature + IHDR + IDAT + IEND
    import zlib

    sig = b'\x89PNG\r\n\x1a\n'

    # IHDR: 1x1, 8-bit grayscale
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0)
    ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xFFFFFFFF)
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + ihdr_crc

    # IDAT: single white pixel (filter byte 0, pixel value 255)
    raw = zlib.compress(b'\x00\xff')
    idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + raw) & 0xFFFFFFFF)
    idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + idat_crc

    # IEND
    iend_crc = struct.pack('>I', zlib.crc32(b'IEND') & 0xFFFFFFFF)
    iend = struct.pack('>I', 0) + b'IEND' + iend_crc

    with open(path, 'wb') as f:
        f.write(sig + ihdr + idat + iend)


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Flask test client with a temp frames directory containing one dummy PNG."""
    frames_dir = tmp_path / 'frames'
    frames_dir.mkdir()
    _make_minimal_png(str(frames_dir / 'frame_00001.png'))

    import firmware_review_tool.app as app_mod
    monkeypatch.setattr(app_mod, 'FRAMES_DIR', str(frames_dir))

    app_mod.app.config['TESTING'] = True
    with app_mod.app.test_client() as client:
        yield client
