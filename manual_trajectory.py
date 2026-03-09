#!/usr/bin/env python3
"""
Manual scroll trajectory for the Hakko FM-203 firmware dump video.

Confirmed by manual inspection of video frames F1–F20070.  Each waypoint is
(frame_number, top_address) where top_address is the address shown on the first
visible row.  None means no hex data is visible (UI transition).

Helper: interpolate_trajectory(frame) → (top_addr, bottom_addr) or None.
"""

# 15 data rows visible on screen; bottom = top + 0x0F0 - 0x10
VISIBLE_ROWS = 15
SCREEN_SPAN = (VISIBLE_ROWS - 1) * 0x10  # 0x0E0

MANUAL_TRAJECTORY = [
    # Pass 1: forward scroll $00000 → $10040
    (1,     0x00000),
    (1728,  0x0FF50),
    # Static at top of ROM
    (5044,  0x0FF50),
    # No hex data visible (UI transition)
    (5045,  None),
    (5229,  None),
    # View restarted at $00000
    (5230,  0x00000),
    (5287,  0x00000),
    # Pass 2: oscillation zone
    (5288,  0x00330),
    (5306,  0x03660),
    (5313,  0x03660),   # paused
    (5314,  0x02FF0),   # reversal → decreasing
    (5319,  0x02330),   # paused
    (5336,  0x02660),
    (5339,  0x02990),   # increasing
    (5367,  0x02CC0),
    (5369,  0x02FF0),
    (5372,  0x03330),
    (5373,  0x03660),
    (5375,  0x03990),
    (5408,  0x04CC0),   # last before reversal
    (5409,  0x04990),   # reversal → decreasing
    (5446,  0x04660),   # decreasing
    (5447,  0x04990),   # reversal → increasing (scroll artifact)
    # Pass 2: steady forward
    (5502,  0x04990),
    (6325,  0x05E70),
    (6660,  0x05E70),   # paused
    (6661,  0x05E70),
    (8610,  0x086A0),
    (8611,  0x086A0),
    (8955,  0x086A0),   # paused
    (8956,  0x08790),
    (11557, 0x0B450),
    (11812, 0x0B450),   # paused
    (11813, 0x0B460),
    (12213, 0x0BFA0),
    (12242, 0x0BFA0),   # paused
    (12243, 0x0BF90),   # reversal → decreasing
    (12299, 0x0BE60),
    (12300, 0x0BE80),   # reversal → increasing
    (12500, 0x0C460),
    (13000, 0x0C630),
    (13500, 0x0D210),
    (14500, 0x0DFA0),
    (15500, 0x0F2B0),
    (16020, 0x0FF90),
    (16083, 0x0FF90),   # paused
    (16084, 0x0FF80),   # reversal → decreasing
    (16106, 0x0FE80),
    (16133, 0x0FE80),   # paused
    (16134, 0x0FE90),   # reversal → increasing
    (19215, 0x13F00),
    (19298, 0x13F00),   # paused at top
    # Fast rewind
    (19299, 0x13F00),
    (19410, 0x00000),
    # Paused at bottom
    (19995, 0x00000),
    # Short forward at end
    (19996, 0x00030),
    (20070, 0x00970),
]


def get_waypoint_spacing(frame):
    """Return the frame span between the two waypoints surrounding `frame`.

    Used to estimate interpolation uncertainty — wider spacing means less
    accurate interpolation.  Returns 0 for exact waypoint matches.
    """
    traj = MANUAL_TRAJECTORY
    if frame < traj[0][0] or frame > traj[-1][0]:
        return None
    prev_f = traj[0][0]
    for i in range(1, len(traj)):
        cur_f = traj[i][0]
        if cur_f >= frame:
            return cur_f - prev_f
        prev_f = cur_f
    return 0


def interpolate_trajectory(frame):
    """Interpolate expected screen position at a given frame.

    Returns (top_addr, bottom_addr) snapped to 0x10 alignment, or None if
    the frame is in a no-data zone.
    """
    traj = MANUAL_TRAJECTORY
    if frame < traj[0][0] or frame > traj[-1][0]:
        return None

    # Find the two surrounding waypoints (prev_wp <= frame <= next_wp)
    prev_f, prev_a = traj[0]
    for i in range(1, len(traj)):
        cur_f, cur_a = traj[i]
        if cur_f >= frame:
            # frame is between prev and cur
            if prev_a is None or cur_a is None:
                return None
            if cur_f == prev_f:
                top = cur_a
            else:
                frac = (frame - prev_f) / (cur_f - prev_f)
                top = prev_a + frac * (cur_a - prev_a)
            top = int(round(top / 0x10)) * 0x10
            return (top, top + SCREEN_SPAN)
        prev_f, prev_a = cur_f, cur_a

    # Past last waypoint
    if prev_a is None:
        return None
    top = (prev_a // 0x10) * 0x10
    return (top, top + SCREEN_SPAN)


if __name__ == '__main__':
    # Quick sanity checks
    test_frames = [1, 1000, 5045, 5100, 5288, 5400, 6419, 10000, 16050,
                   19350, 20000, 20070]
    for f in test_frames:
        result = interpolate_trajectory(f)
        if result is None:
            print(f"  F{f:5d}: no data")
        else:
            top, bot = result
            print(f"  F{f:5d}: ${top:05X}-${bot:05X}")
