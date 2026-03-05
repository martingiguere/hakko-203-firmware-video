#!/usr/bin/env python3
"""Project status diagnostic for Hakko FM-203 firmware extraction pipeline."""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def check_file(path):
    return os.path.exists(path)


def count_pngs(directory):
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith('.png'))


def main():
    print('=== HAKKO FM-203 FIRMWARE PROJECT STATUS ===\n')

    # 1. Video frames
    frames_dir = 'frames'
    if os.path.isdir(frames_dir):
        frames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        print(f'[OK]      Video frames: {len(frames)}')
    else:
        print('[MISSING] frames/ directory')

    # 2. KNN model
    if check_file('fast_knn_classifier.npz'):
        print(f'[OK]      KNN model: {os.path.getsize("fast_knn_classifier.npz") / 1024:.0f} KB')
    else:
        print('[MISSING] fast_knn_classifier.npz')

    # 3. Grid calibration
    if check_file('grid_calibration.json'):
        print('[OK]      Grid calibration')
    else:
        print('[MISSING] grid_calibration.json')

    # 4-5. Firmware text files
    for name, path in [('Extracted firmware', 'extracted_firmware.txt'),
                        ('Merged firmware', 'firmware_merged.txt')]:
        if check_file(path):
            with open(path) as f:
                lines = [l for l in f if l.strip() and not l.startswith('#')]
            print(f'[OK]      {name}: {len(lines)} address lines')
        else:
            print(f'[MISSING] {path}')

    # 6. Crop index
    ci_path = 'crops/crop_index.json'
    ci_addrs = {}
    ref_list = []
    if check_file(ci_path):
        ci = json.load(open(ci_path))
        ci_addrs = {k: v for k, v in ci.items() if isinstance(v, dict)}
        ref_list = ci.get('ref_addresses', [])
        total_frame_refs = sum(len(v.get('frames', [])) for v in ci_addrs.values())
        addrs_with_readings = sum(1 for v in ci_addrs.values() if v.get('readings'))
        print(f'[OK]      Crop index: {len(ci_addrs)} addresses, {total_frame_refs} frame refs, '
              f'{addrs_with_readings} with readings, {len(ref_list)} ref addresses')
    else:
        print('[MISSING] crops/crop_index.json — run precompute.py')

    # 7. Crop images
    if os.path.isdir('crops'):
        crop_dirs = [d for d in os.listdir('crops')
                     if os.path.isdir(os.path.join('crops', d)) and d not in ('gap', 'ref')]
        total_crop_imgs = sum(count_pngs(os.path.join('crops', d)) for d in crop_dirs)
        empty = sum(1 for d in crop_dirs if count_pngs(os.path.join('crops', d)) == 0)
        print(f'[OK]      Crop images: {len(crop_dirs)} addr dirs, {total_crop_imgs} PNGs, {empty} empty dirs')
    else:
        print('[MISSING] crops/ directory')

    # 8. Gap context index
    gap_index_path = 'crops/gap_context_index.json'
    if check_file(gap_index_path):
        gi = json.load(open(gap_index_path))
        print(f'[OK]      Gap context index: {len(gi)} entries')
    else:
        print('[MISSING] crops/gap_context_index.json — run precompute_gaps.py')

    # 9. Gap crops
    gap_count = count_pngs('crops/gap')
    if os.path.isdir('crops/gap'):
        status = '[OK]     ' if gap_count > 0 else '[EMPTY]  '
        print(f'{status} Gap crops: {gap_count} PNGs')
    else:
        print('[MISSING] crops/gap/ directory')

    # 10. Reference crops
    ref_count = count_pngs('crops/ref')
    if os.path.isdir('crops/ref'):
        print(f'[OK]      Ref crops: {ref_count} PNGs')
    else:
        print('[MISSING] crops/ref/ directory')

    # 11. Review state
    if check_file('review_state.json'):
        rs = json.load(open('review_state.json'))
        lines = rs.get('lines', {})
        statuses = {}
        sources = {}
        for info in lines.values():
            s = info.get('status', '?')
            statuses[s] = statuses.get(s, 0) + 1
            src = info.get('source', '?')
            sources[src] = sources.get(src, 0) + 1
        print(f'[OK]      Review state: {len(lines)} lines')
        print(f'          Statuses: {statuses}')
        print(f'          Sources: {sources}')
    else:
        print('[MISSING] review_state.json')

    # 12. Frame moves
    if check_file('frame_moves.json'):
        fm = json.load(open('frame_moves.json'))
        print(f'[OK]      Frame moves: {len(fm)} moves')

    # Coverage
    print('\n=== COVERAGE ===')
    all_addrs = set(f'{a:05X}' for a in range(0, 0x14000, 0x10))
    indexed = set(ci_addrs.keys())
    rom_addrs = set(f'{a:05X}' for a in range(0x04000, 0x14000, 0x10))
    rom_indexed = indexed & rom_addrs
    missing_rom = sorted(rom_addrs - indexed)
    print(f'Full buffer: {len(all_addrs)} lines, indexed: {len(indexed)} ({100 * len(indexed) / len(all_addrs):.1f}%)')
    print(f'ROM (04000-13FFF): {len(rom_addrs)} lines, indexed: {len(rom_indexed)} '
          f'({100 * len(rom_indexed) / len(rom_addrs):.1f}%)')
    print(f'Missing ROM addresses: {len(missing_rom)}')

    if missing_rom:
        gaps = []
        start = prev = None
        for a in missing_rom:
            val = int(a, 16)
            if prev is None or val != prev + 0x10:
                if start is not None:
                    gaps.append((start, prev))
                start = val
            prev = val
        if start is not None:
            gaps.append((start, prev))
        print(f'Missing ROM gap ranges ({len(gaps)} gaps):')
        for s, e in gaps[:15]:
            print(f'  {s:05X}-{e:05X} ({(e - s) // 0x10 + 1} lines)')
        if len(gaps) > 15:
            print(f'  ... and {len(gaps) - 15} more gaps')

    # Consistency checks
    print('\n=== CONSISTENCY ===')
    if ci_addrs:
        missing_pngs = 0
        checked = 0
        for addr, info in list(ci_addrs.items())[:100]:
            addr_dir = os.path.join('crops', addr.lower())
            if not os.path.isdir(addr_dir):
                continue
            actual = set(os.listdir(addr_dir))
            for fr in info.get('frames', []):
                checked += 1
                if f'frame_{fr:05d}.png' not in actual:
                    missing_pngs += 1
        print(f'Sampled 100 addrs, checked {checked} frame refs: {missing_pngs} missing crop PNGs')
    else:
        print('No crop index to check')


if __name__ == '__main__':
    main()
