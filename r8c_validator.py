#!/usr/bin/env python3
"""
R8C instruction validation for OCR-extracted Hakko FM-203 firmware.

Walks the firmware from interrupt vector entry points using BFS,
decoding instruction lengths to verify opcode validity. Flags:
  - Invalid/undefined opcodes (0x01)
  - Jump targets outside ROM
  - Desync (instruction walk falls off valid region)
  - UND (0xFF) instructions in code paths

Usage:
    python3 r8c_validator.py [firmware_merged.txt]              # report only
    python3 r8c_validator.py --flag-review                      # + flag in review_state.json
    python3 r8c_validator.py --fix                              # reserved (not implemented)
"""

import sys
import os
import json
from collections import deque

from r8c_opcode_table import get_instruction_length

# R8C/25 (R5F21258SNFP) memory layout
ROM_RANGES = [(0x04000, 0x0FFFF), (0x10000, 0x13FFF)]
VECTOR_TABLE_START = 0x0FF00
VECTOR_TABLE_END = 0x0FFFB
RESET_VECTOR_ADDR = 0x0FFFC  # 3 bytes (byte at 0xFFFF is OPT)
DEFAULT_HANDLER = 0x0FC4E     # most unused vectors point here


def _sign_ext(val, bits):
    """Sign-extend a value from given bit width."""
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def _in_rom(addr):
    """Check if address falls within ROM ranges."""
    return any(lo <= addr <= hi for lo, hi in ROM_RANGES)


def load_firmware(path='firmware_merged.txt'):
    """Load firmware hex dump as a bytearray indexed by address.

    Returns bytearray of size 0x14000 ($00000-$13FFF), with
    unspecified addresses filled with 0xFF (erased flash state).
    """
    fw = bytearray(b'\xFF' * 0x14000)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            addr_str, rest = line.split(':', 1)
            # Strip [REF] or other tags
            data_part = rest.split('[')[0].strip()
            try:
                addr = int(addr_str.strip(), 16)
            except ValueError:
                continue
            for i, bh in enumerate(data_part.split()):
                if addr + i < len(fw):
                    try:
                        fw[addr + i] = int(bh, 16)
                    except ValueError:
                        pass
    return fw


def extract_vectors(fw):
    """Extract interrupt vector addresses from the vector table.

    Returns list of (name, address) tuples for non-erased, non-default vectors
    plus the reset vector.
    """
    vectors = []

    # Reset vector: 3 bytes at $0FFFC (little-endian 20-bit)
    reset_addr = fw[0xFFFC] | (fw[0xFFFD] << 8) | ((fw[0xFFFE] & 0x0F) << 16)
    vectors.append(('reset', reset_addr))

    # Fixed vector table names ($0FFDC-$0FFFB)
    fixed_names = {
        0x0FFDC: 'UND_intr', 0x0FFE0: 'overflow', 0x0FFE4: 'BRK_intr',
        0x0FFE8: 'addr_match', 0x0FFEC: 'single_step',
        0x0FFF0: 'watchdog', 0x0FFF4: 'DBC', 0x0FFF8: 'NMI',
    }

    # Scan all 4-byte vector entries from $0FF00 to $0FFFB
    for base in range(VECTOR_TABLE_START, VECTOR_TABLE_END + 1, 4):
        addr = fw[base] | (fw[base + 1] << 8) | ((fw[base + 2] & 0x0F) << 16)
        if addr == 0xFFFFF:  # erased (all FF)
            continue
        if addr == DEFAULT_HANDLER:
            continue
        name = fixed_names.get(base, f'vec_{base:05X}')
        vectors.append((name, addr))

    # Deduplicate by address, keep first name
    seen = set()
    unique = []
    for name, addr in vectors:
        if addr not in seen:
            seen.add(addr)
            unique.append((name, addr))
    return unique


def walk_code(fw, entry_points):
    """BFS instruction walk from entry points.

    Returns:
        visited: set of instruction start addresses successfully decoded
        issues: list of (addr, message) strings
    """
    visited = set()
    issues = []
    queue = deque()

    for name, addr in entry_points:
        queue.append((name, addr))

    while queue:
        origin_name, addr = queue.popleft()

        if addr in visited:
            continue
        if not _in_rom(addr):
            issues.append((addr, f"Entry '{origin_name}' at ${addr:05X} is outside ROM"))
            continue

        # Linear disassembly from this address
        while True:
            if addr in visited:
                break
            if not _in_rom(addr):
                issues.append((addr, f"Execution fell outside ROM at ${addr:05X} (from {origin_name})"))
                break

            b1 = fw[addr]
            length = get_instruction_length(fw, addr)

            if length == 0:
                issues.append((addr, f"Invalid opcode 0x{b1:02X} at ${addr:05X}"))
                break

            # Check instruction doesn't extend past ROM
            end = addr + length - 1
            if not _in_rom(end):
                issues.append((addr, f"Instruction at ${addr:05X} extends past ROM end"))
                break

            visited.add(addr)

            # --- Control flow analysis ---

            # Returns: end this path
            if b1 == 0xF3:  # RTS
                break
            if b1 == 0xFB:  # REIT (return from interrupt)
                break
            if b1 == 0x7D and fw[addr + 1] == 0xF2:  # EXITD
                break
            if b1 == 0x00:  # BRK (software interrupt)
                break

            # UND: flag and stop
            if b1 == 0xFF:
                issues.append((addr, f"UND instruction at ${addr:05X}"))
                break

            # Unconditional jumps: follow target, no fall-through
            if 0x60 <= b1 <= 0x67:  # JMP.S
                dsp = b1 & 0x07
                target = addr + 2 + dsp  # R8C: base=2 for short branches
                _enqueue_target(queue, target, f"JMP.S@${addr:05X}", issues)
                break

            if b1 == 0xFE:  # JMP.B
                dsp = _sign_ext(fw[addr + 1], 8)
                target = addr + 2 + dsp
                _enqueue_target(queue, target, f"JMP.B@${addr:05X}", issues)
                break

            if b1 == 0xF4:  # JMP.W
                dsp = _sign_ext(fw[addr + 1] | (fw[addr + 2] << 8), 16)
                target = addr + 3 + dsp
                _enqueue_target(queue, target, f"JMP.W@${addr:05X}", issues)
                break

            if b1 == 0xFC:  # JMP.A
                target = fw[addr + 1] | (fw[addr + 2] << 8) | ((fw[addr + 3] & 0x0F) << 16)
                _enqueue_target(queue, target, f"JMP.A@${addr:05X}", issues)
                break

            # Indirect jumps via 0x7D prefix: can't determine target
            if b1 == 0x7D:
                b2 = fw[addr + 1]
                upper = (b2 >> 4) & 0x0F
                if upper == 0x0 or upper == 0x2:  # JMPI.A / JMPI.W
                    break  # can't follow indirect
                if upper == 0x1 or upper == 0x3:  # JSRI.A / JSRI.W (indirect call)
                    addr += length
                    continue

            # Conditional branches: follow both paths
            if 0x68 <= b1 <= 0x6F:  # JCnd:S
                dsp = _sign_ext(fw[addr + 1], 8)
                target = addr + 2 + dsp
                _enqueue_target(queue, target, f"JCnd@${addr:05X}", issues)
                addr += length
                continue

            if b1 == 0x7D and ((fw[addr + 1] >> 4) & 0x0F) == 0xC:  # JCnd:B
                dsp = _sign_ext(fw[addr + 2], 8)
                target = addr + 3 + dsp
                _enqueue_target(queue, target, f"JCnd:B@${addr:05X}", issues)
                addr += length
                continue

            # Calls: follow target AND fall through
            if b1 == 0xF5:  # JSR.W
                dsp = _sign_ext(fw[addr + 1] | (fw[addr + 2] << 8), 16)
                target = addr + 3 + dsp
                _enqueue_target(queue, target, f"JSR.W@${addr:05X}", issues)
                addr += length
                continue

            if b1 == 0xFD:  # JSR.A
                target = fw[addr + 1] | (fw[addr + 2] << 8) | ((fw[addr + 3] & 0x0F) << 16)
                _enqueue_target(queue, target, f"JSR.A@${addr:05X}", issues)
                addr += length
                continue

            # ADJNZ/SBJNZ: conditional loop branches
            if b1 in (0xF8, 0xF9, 0xFA):
                label = fw[addr + length - 1]
                dsp = _sign_ext(label, 8)
                target = addr + length + dsp
                _enqueue_target(queue, target, f"ADJNZ@${addr:05X}", issues)
                addr += length
                continue

            # All other instructions: simple fall-through
            addr += length

    return visited, issues


def _enqueue_target(queue, target, origin, issues):
    """Add a jump/branch target to the walk queue, flagging if outside ROM."""
    if _in_rom(target):
        queue.append((origin, target))
    else:
        issues.append((target, f"Jump target ${target:05X} outside ROM (from {origin})"))


def validate_vectors(fw):
    """Check the vector table for consistency."""
    issues = []

    # Check default vectors are consistent
    default_count = 0
    non_default = []
    erased = []

    for base in range(VECTOR_TABLE_START, VECTOR_TABLE_END + 1, 4):
        addr = fw[base] | (fw[base + 1] << 8) | ((fw[base + 2] & 0x0F) << 16)
        if addr == 0xFFFFF:
            erased.append(base)
        elif addr == DEFAULT_HANDLER:
            default_count += 1
        else:
            non_default.append((base, addr))

    # Reset vector
    reset = fw[0xFFFC] | (fw[0xFFFD] << 8) | ((fw[0xFFFE] & 0x0F) << 16)
    opt_byte = fw[0xFFFF]

    results = {
        'reset_vector': reset,
        'opt_byte': opt_byte,
        'default_handler': DEFAULT_HANDLER,
        'default_count': default_count,
        'erased_count': len(erased),
        'non_default': non_default,
    }

    # Check non-default vectors point to valid ROM
    for base, addr in non_default:
        if not _in_rom(addr):
            issues.append((base, f"Vector at ${base:05X} -> ${addr:05X} is outside ROM"))

    # Check reset vector is in ROM
    if not _in_rom(reset):
        issues.append((0xFFFC, f"Reset vector ${reset:05X} is outside ROM"))

    return results, issues


def validate_firmware(path='firmware_merged.txt'):
    """Run all validations and return a report dict."""
    print(f"Loading firmware from {path}...")
    fw = load_firmware(path)

    print("Analyzing vector table...")
    vec_info, vec_issues = validate_vectors(fw)

    print(f"  Reset vector: ${vec_info['reset_vector']:05X}")
    print(f"  OPT byte: 0x{vec_info['opt_byte']:02X}")
    print(f"  Default handler: ${vec_info['default_handler']:05X}")
    print(f"  Default vectors: {vec_info['default_count']}")
    print(f"  Erased vectors: {vec_info['erased_count']}")
    print(f"  Non-default vectors: {len(vec_info['non_default'])}")
    for base, addr in vec_info['non_default']:
        print(f"    ${base:05X} -> ${addr:05X}")

    # Extract entry points
    entries = extract_vectors(fw)
    print(f"\nEntry points for code walk ({len(entries)}):")
    for name, addr in entries:
        print(f"  {name}: ${addr:05X}")

    # Also walk the default handler
    entries.append(('default_handler', DEFAULT_HANDLER))

    print(f"\nWalking code from {len(entries)} entry points...")
    visited, walk_issues = walk_code(fw, entries)

    # Deduplicate issues (same addr+message from different paths)
    seen_issues = set()
    all_issues = []
    for addr, msg in vec_issues + walk_issues:
        key = (addr, msg)
        if key not in seen_issues:
            seen_issues.add(key)
            all_issues.append((addr, msg))

    # Compute coverage statistics
    rom_bytes = sum(hi - lo + 1 for lo, hi in ROM_RANGES)
    code_bytes = 0
    for addr in visited:
        length = get_instruction_length(fw, addr)
        if length > 0:
            code_bytes += length

    # Find all-FF regions within visited code (potential OCR errors)
    ff_instructions = []
    for addr in sorted(visited):
        b1 = fw[addr]
        if b1 == 0xFF:
            ff_instructions.append(addr)

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Instructions decoded:  {len(visited)}")
    print(f"Code bytes covered:    {code_bytes} / {rom_bytes} ROM bytes")
    print(f"Code coverage:         {100*code_bytes/rom_bytes:.1f}%")
    print(f"Issues found:          {len(all_issues)}")

    if all_issues:
        print(f"\n--- Issues ({len(all_issues)}) ---")
        for addr, msg in sorted(all_issues):
            print(f"  ${addr:05X}: {msg}")

    # Build report
    report = {
        'firmware_path': path,
        'vectors': vec_info,
        'instructions_decoded': len(visited),
        'code_bytes': code_bytes,
        'rom_bytes': rom_bytes,
        'coverage_pct': round(100 * code_bytes / rom_bytes, 1),
        'issues': [(f"${a:05X}", m) for a, m in sorted(all_issues)],
        'issue_count': len(all_issues),
    }

    return report, visited, all_issues


def save_report(report, path='validation_report.txt'):
    """Save validation report to text file."""
    with open(path, 'w') as f:
        f.write("R8C Instruction Validation Report\n")
        f.write(f"Firmware: {report['firmware_path']}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"Reset vector:        ${report['vectors']['reset_vector']:05X}\n")
        f.write(f"OPT byte:            0x{report['vectors']['opt_byte']:02X}\n")
        f.write(f"Default handler:     ${report['vectors']['default_handler']:05X}\n")
        f.write(f"Default vectors:     {report['vectors']['default_count']}\n")
        f.write(f"Non-default vectors: {len(report['vectors']['non_default'])}\n")
        for base, addr in report['vectors']['non_default']:
            f.write(f"  ${base:05X} -> ${addr:05X}\n")

        f.write(f"\nInstructions decoded: {report['instructions_decoded']}\n")
        f.write(f"Code bytes covered:   {report['code_bytes']} / {report['rom_bytes']}\n")
        f.write(f"Code coverage:        {report['coverage_pct']}%\n")

        f.write(f"\nIssues: {report['issue_count']}\n")
        for addr_str, msg in report['issues']:
            f.write(f"  {addr_str}: {msg}\n")

    print(f"\nReport saved to {path}")


def flag_review_state(issues, review_state_path='review_state.json'):
    """Flag addresses with validation issues in review_state.json.

    Only changes status to "flagged" for addresses that aren't already
    "accepted" or "edited" (preserves manual review work).
    """
    if not os.path.exists(review_state_path):
        print(f"  {review_state_path} not found — skipping")
        return 0

    with open(review_state_path) as f:
        state = json.load(f)

    lines = state.get('lines', {})
    flagged_count = 0

    for addr_int, msg in issues:
        # Convert instruction address to line address (align to 0x10)
        line_addr = addr_int & 0xFFFF0
        addr_key = f"{line_addr:05X}"

        if addr_key not in lines:
            continue

        line = lines[addr_key]
        # Don't override manual review work
        if line.get('status') in ('accepted', 'edited'):
            continue

        if line.get('status') != 'flagged':
            line['status'] = 'flagged'
            flagged_count += 1

    if flagged_count > 0:
        import datetime
        state['last_saved'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(review_state_path, 'w') as f:
            json.dump(state, f)
        print(f"  Flagged {flagged_count} address lines in {review_state_path}")
    else:
        print(f"  No new lines to flag in {review_state_path}")

    return flagged_count


def main():
    args = sys.argv[1:]
    flag_review = '--flag-review' in args
    do_fix = '--fix' in args
    args = [a for a in args if not a.startswith('--')]
    path = args[0] if args else 'firmware_merged.txt'

    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)

    if do_fix:
        print("ERROR: --fix is not implemented yet.")
        print("       The validator only flags addresses for manual review.")
        sys.exit(1)

    report, visited, issues = validate_firmware(path)
    save_report(report)

    if flag_review:
        print("\nFlagging issues in review_state.json...")
        flag_review_state(issues)


if __name__ == '__main__':
    main()
