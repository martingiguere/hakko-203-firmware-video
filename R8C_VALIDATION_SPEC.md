# R8C/Tiny Instruction Validation Spec for OCR Firmware Verification

## Purpose

Use R8C/Tiny instruction set knowledge to detect OCR errors in the extracted firmware by validating that byte sequences form legal instructions. Flag addresses with invalid opcodes or implausible instruction patterns for manual review.

## Architecture Background

- **CPU**: R5F21258SNFP (R8C/24 Group, R8C/Tiny Series)
- **Instruction set**: 89 instructions, variable-length 1–5 bytes
- **Encoding formats**: :G (generic 2-byte opcode), :Q (quick), :S (short 1-byte), :Z (zero)
- **Only undefined first byte**: `$01`
- **Reset vector**: `$0FFFC-$0FFFF` → entry point `$0FBAE`
- **ROM code region**: `$04990-$107E0` (real code + data tables)
- **Data tables**: `$10000-$107E0` (calibration ramps, lookup tables — not code)

## Available References

| Resource | Location | Content |
|----------|----------|---------|
| Instruction encoding doc | `../R5F21258SNFP_emulator/INSTRUCTION_ENCODING.md` | Complete first-byte decode map (section 6), all 256 values mapped |
| Reference implementation analysis | `../R5F21258SNFP_emulator/REFERENCE_IMPL.md` | GNU binutils R8C simulator analysis (~3000 lines C) |
| Emulator spec | `../R5F21258SNFP_emulator/SPECIFICATION.md` | CPU registers, flags, memory map |
| R8C Software Manual PDF | `../R5F21258SNFP_emulator/r8c-tiny_series_Software_Manual.pdf` | Official instruction reference |
| binja-m16c decoder | `github.com/whitequark/binja-m16c` | Python M16C decoder, **0BSD license** (freely reusable) |

## Implementation Plan

### Phase 1: Instruction Length Table (`r8c_opcode_table.py`)

Build a Python lookup table mapping each first byte (0x00-0xFF) to its minimum instruction length. Source: `INSTRUCTION_ENCODING.md` section 6.

```python
# Returns (min_length, max_length, mnemonic, needs_byte2)
# min_length: minimum bytes this instruction consumes
# max_length: maximum bytes (depends on addressing mode in byte2)
# needs_byte2: True if byte2 determines final length (for :G format)
OPCODE_TABLE = {
    0x00: (1, 1, 'BRK', False),
    0x01: (0, 0, 'UNDEFINED', False),  # only undefined byte
    0x02: (2, 2, 'MOV.B:S R0L,dsp8[SB]', False),
    0x04: (1, 1, 'NOP', False),
    # ... all 256 entries from encoding doc section 6
    0x76: (2, 5, 'ALU.B:G prefix', True),  # byte2 selects instruction
    0x77: (2, 5, 'ALU.W:G prefix', True),
    # etc.
}
```

For :G format instructions (byte1 like `0x76`, `0x77`, `0x78`, `0x79`, `0x7A`, `0x7B`, `0x7C`, `0x7D`, `0xEB`), the second byte determines the addressing mode which determines the total length:
- Register direct (codes 0-7): +0 extra bytes
- dsp8 (codes 8-B): +1 byte
- dsp16 (codes C-E): +2 bytes
- abs16 (code F): +2 bytes
- Plus immediate if applicable

### Phase 2: Firmware Walker (`r8c_validator.py`)

Walk the firmware from known entry points, computing instruction boundaries.

```python
def validate_firmware(firmware_bytes, entry_points):
    """Walk firmware from entry points, flag invalid sequences.

    Args:
        firmware_bytes: dict of addr -> [16 bytes] from firmware_merged.txt
        entry_points: list of starting addresses (reset vector, interrupt vectors)

    Returns:
        List of (addr, issue_type, details) for flagged addresses.
    """
```

**Entry points to trace from:**
1. Reset vector at `$0FBAE` (primary entry)
2. Interrupt vectors at `$0FFDC-$0FFFF` (each 4-byte entry has a 20-bit target address)
3. Any `JSR`/`JMP` target addresses discovered during tracing

**Walking algorithm:**
1. Start at entry point
2. Look up first byte in `OPCODE_TABLE`
3. If undefined (`$01`): flag address, stop this path
4. Compute instruction length from byte1 (and byte2 if :G format)
5. Advance PC by instruction length
6. If instruction is a branch/jump: add target to entry points queue
7. If instruction is `RTS`/`REIT`: stop this path
8. Continue until hitting already-visited address or end of ROM

**What to flag:**
- `INVALID_OPCODE`: first byte is `$01` (undefined)
- `JUMP_OUTSIDE_ROM`: JMP/JSR target lands outside `$04990-$107E0`
- `DESYNC`: instruction length would cross a 16-byte address boundary in an implausible way
- `UNREACHABLE`: code region not reachable from any entry point (might be data, not an error)

### Phase 3: Pattern-Based Checks (no disassembly needed)

These can run independently without the full instruction walker.

**3a. Vector table validation:**
- `$0FF70-$0FFCF`: should be `4E FC 00 00` repeated (24 times)
- `$0FFDC-$0FFF3`: should be `4E FC 00 xx` (programmed vectors with ISP ID bytes)
- Flag any deviation

**3b. Data table monotonicity:**
- `$10000-$107E0`: calibration ramp tables with monotonically non-decreasing values
- Detect reversals within known ramp sequences
- Known ramps: `$10028-$1006F` (values: 00, 06, 08, 0A, 0C, 0E, 10, 12, 14)

**3c. Jump target validation:**
- `$F5` byte = `JMP.B` (2 bytes: opcode + signed 8-bit displacement)
- Target = current_addr + 2 + displacement
- Flag if target outside ROM code region

**3d. Common OCR error detection:**
- `$F1` appearing in mostly-FF regions: likely `FF` misread as `F1`
- `$08` in data tables where context suggests `$05`: 8↔5 confusion
- High density of `$4E FC` outside vector table: likely misassigned vector table frames

### Phase 4: Integration

Add as a pipeline step or standalone validation script:

```
post_steps = [
    ...existing steps...
    ('R8C instruction validation',       'r8c_validator.py'),
]
```

Output: `validation_report.txt` listing flagged addresses with issue types. Optionally auto-flag in `review_state.json` for manual review.

## Expected Impact

- **Vector table check**: catches any OCR corruption in the 96 bytes at `$0FF70-$0FFFF`
- **Instruction walker**: traces ~40-60KB of reachable code, flags any invalid opcode encountered
- **Jump target validation**: catches corrupted displacement bytes in branch instructions
- **Data table monotonicity**: catches 8↔5 confusion in calibration ramps (~200 bytes)
- **Limitation**: one wrong byte desyncs the variable-length walker; it can only flag the first error per trace path before losing sync

## Dependencies

- `firmware_merged.txt` (or `review_state.json` for live data)
- `INSTRUCTION_ENCODING.md` section 6 (first-byte decode map) for building the opcode table
- No external libraries needed — pure Python
