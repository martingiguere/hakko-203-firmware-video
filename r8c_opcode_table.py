#!/usr/bin/env python3
"""
R8C/Tiny instruction length lookup table.

Source: INSTRUCTION_ENCODING.md (R8C/Tiny Series Software Manual REJ09B0001-0200)
Provides get_instruction_length() for walking through R8C machine code.

Encoding formats:
  :G  Generic  2-byte opcode (byte1 + byte2) with 4-bit SRC/DEST fields
  :Q  Quick    2-byte opcode with 4-bit immediate + 4-bit DEST in byte2
  :S  Short    1-byte opcode with 3-bit DEST or 2-bit SRC embedded
  :Z  Zero     1-byte opcode, implicit #0 immediate
"""


def _operand_extra(code):
    """Extra bytes for a 4-bit standard operand code (Section 2.1).

    0-7: register direct -> 0
    8-B: dsp:8[base]     -> 1
    C-E: dsp:16[base]    -> 2
      F: abs16           -> 2
    """
    if code <= 0x7:
        return 0
    if code <= 0xB:
        return 1
    return 2


def _short_dest_extra(ddd):
    """Extra bytes for 3-bit short DEST code (Section 2.2).

    011(3),100(4): register -> 0
    101(5),110(6): dsp8     -> 1
    111(7):        abs16    -> 2
    """
    if ddd <= 4:
        return 0
    if ddd <= 6:
        return 1
    return 2


def get_instruction_length(data, offset):
    """Return instruction length in bytes at given offset, or 0 if invalid.

    Args:
        data: bytes, bytearray, or list of ints
        offset: index of first byte of instruction
    """
    if offset >= len(data):
        return 0

    b1 = data[offset]

    # --- 0x00: BRK (1 byte) ---
    if b1 == 0x00:
        return 1

    # --- 0x01: undefined (only truly invalid first byte) ---
    if b1 == 0x01:
        return 0

    # --- 0x02-0x3F: :S format (AND/OR/ADD/SUB/MOV/CMP src,dest) ---
    # Low 2 bits encode operand: 00=reg(1), 01=dsp8(2), 10=dsp8(2), 11=abs16(3)
    # Exception: 0x04 = NOP (also has low2=00, returns 1 correctly)
    if b1 <= 0x3F:
        low2 = b1 & 0x03
        if low2 == 0:
            return 1
        if low2 <= 2:
            return 2
        return 3

    # --- 0x40-0x5F: Bit ops :S (BCLR/BSET/BNOT/BTST) + dsp8 ---
    if b1 <= 0x5F:
        return 2

    # --- 0x60-0x67: JMP.S (3-bit displacement, 1 byte) ---
    if b1 <= 0x67:
        return 1

    # --- 0x68-0x6F: JCnd:S + dsp8 (2 bytes) ---
    if b1 <= 0x6F:
        return 2

    # --- 0x70-0x7F: Multi-byte opcode prefixes ---
    if b1 <= 0x7F:
        return _decode_7x(data, offset)

    # --- 0x80-0x9F: :S #IMM8,dest and :G src,dest ---
    if b1 <= 0x9F:
        return _decode_80_9f(data, offset)

    # --- 0xA0-0xBF: INC/DEC/MOV:Z/NOT:S and :G src,dest ---
    if b1 <= 0xBF:
        return _decode_a0_bf(data, offset)

    # --- 0xC0-0xDF: MOV:S #IMM8, :Q, CMP:G ---
    if b1 <= 0xDF:
        return _decode_c0_df(data, offset)

    # --- 0xE0-0xFF: CMP:S, system ops, jumps ---
    return _decode_e0_ff(data, offset)


# ---- Internal dispatch helpers ----

def _g_src_dest_len(data, offset):
    """Length for :G two-operand src,dest (byte2 = SRC_DEST)."""
    if offset + 1 >= len(data):
        return 0
    b2 = data[offset + 1]
    return 2 + _operand_extra((b2 >> 4) & 0x0F) + _operand_extra(b2 & 0x0F)


def _q_dest_len(data, offset):
    """Length for :Q format (byte2 = IMM4_DEST)."""
    if offset + 1 >= len(data):
        return 0
    return 2 + _operand_extra(data[offset + 1] & 0x0F)


def _decode_7x(data, offset):
    """Decode 0x70-0x7F prefix group."""
    if offset + 1 >= len(data):
        return 0

    b1 = data[offset]
    b2 = data[offset + 1]
    upper = (b2 >> 4) & 0x0F
    lower = b2 & 0x0F

    # 0x70/0x71: MOVA or similar — :G src,dest format
    if b1 <= 0x71:
        return _g_src_dest_len(data, offset)

    # 0x72/0x73: MOV.size:G src,dest
    if b1 <= 0x73:
        return _g_src_dest_len(data, offset)

    # 0x74: MOVDir (2 bytes)
    if b1 == 0x74:
        return 2

    # 0x75: LDC/STC group
    if b1 == 0x75:
        # bit 7: 1=LDC src,creg  0=STC creg,dest
        return 2 + _operand_extra(lower)

    # 0x76/0x77: ALU :G (AND#/ADD#/SUB#/ADC#/SBB#/CMP#/MOV#/OR#/DIV/DIVU/DIVX/ADCF/ABS)
    if b1 <= 0x77:
        imm_size = 2 if b1 == 0x77 else 1
        dest_extra = _operand_extra(lower)
        # Upper nibble selects operation
        if upper in (0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x7, 0x8, 0xA, 0xC):
            return 2 + dest_extra + imm_size   # #IMM,dest
        if upper in (0x9, 0xD):
            return 2 + _operand_extra(lower)   # unary src (DIV/DIVX)
        if upper == 0xE:
            if lower in (0x0, 0x1, 0x3):       # DIV/DIVU/DIVX #IMM
                return 2 + imm_size
            return 2 + dest_extra              # ADCF dest
        if upper == 0xF:
            return 2 + dest_extra              # ABS dest
        return 2 + dest_extra                  # fallback (0x3, 0xB)

    # 0x78/0x79: MUL group
    if b1 <= 0x79:
        imm_size = 2 if b1 == 0x79 else 1
        if upper in (0x4, 0x5):
            return 2 + _operand_extra(lower) + imm_size  # MUL/MULU #IMM
        if upper in (0x0, 0x1):
            return _g_src_dest_len(data, offset)          # MUL/MULU src,dest
        return 2 + _operand_extra(lower)

    # 0x7A: LDE/STE group
    if b1 == 0x7A:
        mode = (b2 >> 4) & 0x07  # bits [6:4]
        if mode in (0, 1, 4, 5):
            return 2 + _operand_extra(lower) + 3   # +abs20/dsp20
        return 2 + _operand_extra(lower)           # [A1A0] or other

    # 0x7B: MULU/PUSH/XCHG group
    if b1 == 0x7B:
        return 2 + _operand_extra(lower)

    # 0x7C/0x7D: Bit ops and special
    if b1 <= 0x7D:
        return _decode_7c_7d(data, offset)

    # 0x7E/0x7F: verify — treat as :G src,dest
    return _g_src_dest_len(data, offset)


def _decode_7c_7d(data, offset):
    """Decode 0x7C/0x7D prefix (bit operations, BCD, jumps, special)."""
    if offset + 1 >= len(data):
        return 0

    b1 = data[offset]
    b2 = data[offset + 1]
    upper = (b2 >> 4) & 0x0F
    lower = b2 & 0x0F
    is_word = (b1 == 0x7D)
    imm_size = 2 if is_word else 1

    # === 0x7D-only overrides ===
    if b1 == 0x7D:
        if upper <= 0x3:               # JMPI.A(0), JSRI.A(1), JMPI.W(2), JSRI.W(3)
            return 2 + _operand_extra(lower)
        if b2 & 0xF8 == 0xA0:         # LDIPL #IMM3 (byte2=1010_0xxx)
            return 2
        if upper == 0xB:               # ADD.Q #IMM4,SP
            return 2
        if upper == 0xC:               # JCnd:B + dsp8
            return 3
        if b2 == 0xF2:                 # EXITD
            return 2

    # === 0x7C-only overrides ===
    if b1 == 0x7C:
        if b2 == 0xF0:                 # LDCTX abs16,abs20
            return 7
        if b2 == 0xF2:                 # ENTER #IMM8
            return 3
        if b2 == 0xF3:                 # EXTS.W R0
            return 2

    # === BCD register forms (both): byte2=0110_1100..0110_1111 ===
    if upper == 0x6 and lower >= 0xC:
        return 2                       # DADD/DSUB/DADC/DSBB reg form

    # === Common bit/logic operations ===
    if upper <= 0x1:                   # BTSTC(0), BTSTS(1)
        return 2 + _operand_extra(lower)
    if upper == 0x2:                   # BMCnd dest + condition byte
        return 2 + _operand_extra(lower) + 1
    if upper <= 0x7:                   # BNTST(3),BAND(4),BNAND(5),BOR(6),BNOR(7)
        return 2 + _operand_extra(lower)
    if upper <= 0xA:                   # BCLR:G(8), BSET:G(9), BNOT:G(A)
        return 2 + _operand_extra(lower)
    if upper == 0xB:                   # BTST:G (0x7D handled above as ADD.Q)
        return 2 + _operand_extra(lower)
    if upper == 0xC:                   # BXOR (0x7D handled above as JCnd:B)
        return 2 + _operand_extra(lower)
    if upper == 0xD:                   # BNXOR / BMCnd C
        return 2 + _operand_extra(lower)
    if upper == 0xE:                   # ADD#SP, BCD #IMM
        if lower == 0xB:
            return 2 + imm_size        # ADD #IMM,SP
        if lower >= 0xC:
            return 2 + imm_size        # DADD/DSUB/DADC/DSBB #IMM
        return 2 + _operand_extra(lower)
    if upper == 0xF:
        return 2                       # unknown F-pattern fallback

    return 2


def _decode_80_9f(data, offset):
    """Decode 0x80-0x9F: :S #IMM8,dest (ADD/SUB/AND/OR) and :G src,dest."""
    b1 = data[offset]
    ddd = b1 & 0x07

    # DDD 0-2: :G two-operand src,dest (e.g. AND.B:G at 0x90)
    if ddd <= 2:
        return _g_src_dest_len(data, offset)

    # DDD 3-7: :S #IMM8,dest
    return 1 + _short_dest_extra(ddd) + 1


def _decode_a0_bf(data, offset):
    """Decode 0xA0-0xBF: INC/DEC/MOV:Z/NOT:S and :G src,dest."""
    b1 = data[offset]
    ddd = b1 & 0x07

    # DDD 0-1: :G two-operand (ADD:G, ADC:G etc.)
    if ddd <= 1:
        return _g_src_dest_len(data, offset)

    # DDD 2: 1-byte instructions (INC.W A0=0xB2, DEC.W A0/A1, etc.)
    if ddd == 2:
        return 1

    # DDD 3-7: :S no-immediate (INC/DEC/MOV:Z/NOT)
    return 1 + _short_dest_extra(ddd)


def _decode_c0_df(data, offset):
    """Decode 0xC0-0xDF: CMP:G, MOV:S #IMM8, :Q formats."""
    b1 = data[offset]
    ddd = b1 & 0x07

    # 0xC0/0xC1: CMP.size:G src,dest
    if b1 <= 0xC1:
        return _g_src_dest_len(data, offset)
    # 0xC2: uncertain — treat as :G
    if b1 == 0xC2:
        return _g_src_dest_len(data, offset)
    # 0xC3-0xC7: MOV.B:S #IMM8,dest
    if b1 <= 0xC7:
        return 1 + _short_dest_extra(ddd) + 1

    # 0xC8-0xDF: :Q format (ADD:Q, CMP:Q, MOV:Q, SHL:Q, SHA:Q, ROT:Q etc.)
    return _q_dest_len(data, offset)


def _decode_e0_ff(data, offset):
    """Decode 0xE0-0xFF: CMP:S, system ops, jumps, returns."""
    b1 = data[offset]
    ddd = b1 & 0x07

    # 0xE0-0xE2: uncertain — treat as :G src,dest
    if b1 <= 0xE2:
        return _g_src_dest_len(data, offset)
    # 0xE3-0xE7: CMP.B:S #IMM8,dest
    if b1 <= 0xE7:
        return 1 + _short_dest_extra(ddd) + 1

    if b1 == 0xE8: return 2    # STNZ #IMM8
    if b1 == 0xE9: return 2    # STZ #IMM8
    if b1 == 0xEA: return 3    # STZX #IMM1,#IMM2

    # 0xEB: multi-instruction prefix
    if b1 == 0xEB:
        return _decode_eb(data, offset)

    # 0xEC-0xEF: PUSH/POP/PUSHM/POPM (2 bytes)
    if b1 <= 0xEF:
        return 2

    # 0xF0/0xF1: :G format (verify)
    if b1 <= 0xF1:
        return _g_src_dest_len(data, offset)
    # 0xF2: uncertain instruction (2 bytes)
    if b1 == 0xF2:
        return 2

    if b1 == 0xF3: return 1    # RTS
    if b1 == 0xF4: return 3    # JMP.W + dsp16
    if b1 == 0xF5: return 3    # JSR.W + dsp16
    if b1 == 0xF6: return 1    # INTO
    if b1 == 0xF7: return 1    # uncertain (possibly SMOVF or WAIT)

    # 0xF8/0xF9: ADJNZ.B/ADJNZ.W, 0xFA: SBJNZ.B
    # Format: byte1 + byte2(IMM4_DEST) + dest_operands + label8
    if b1 <= 0xFA:
        if offset + 1 >= len(data):
            return 0
        dest = data[offset + 1] & 0x0F
        return 3 + _operand_extra(dest)

    if b1 == 0xFB: return 1    # REIT (return from interrupt)

    if b1 == 0xFC: return 4    # JMP.A + abs20
    if b1 == 0xFD: return 4    # JSR.A + abs20
    if b1 == 0xFE: return 2    # JMP.B + dsp8
    return 1                   # 0xFF: UND


def _decode_eb(data, offset):
    """Decode 0xEB multi-instruction prefix."""
    if offset + 1 >= len(data):
        return 0

    b2 = data[offset + 1]

    # INT #IMM: byte2 >= 0xC0 (11_IIIIII)
    if b2 >= 0xC0:
        return 2
    # LDINTB: byte2 = 0x20, 8 bytes total
    if b2 == 0x20:
        return 8
    # LDC #IMM16: byte2 = 0DDD_0000
    if (b2 & 0x8F) == 0x00:
        return 4
    # FSET: byte2 = 0DDD_0100
    if (b2 & 0x8F) == 0x04:
        return 2
    # FCLR: byte2 = 0DDD_0101
    if (b2 & 0x8F) == 0x05:
        return 2
    # STC: byte2 = 0DDD_0001 (verify — may need dest operand bytes)
    if (b2 & 0x8F) == 0x01:
        return 2

    return 2  # fallback for other 0xEB patterns
