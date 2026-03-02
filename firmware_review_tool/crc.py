"""Checksum utility and firmware binary builder."""

BASE_ADDR = 0x00000
BUFFER_SIZE = 81920  # 0x00000 to 0x13FFF inclusive = 80 KB
ROM_START = 0x04000
ROM_SIZE = 65536  # 0x04000 to 0x13FFF = 64 KB
EXPECTED_CHECKSUM = 0x00D2F2FF


def compute_byte_sum_32(firmware_bytes):
    """Compute 32-bit byte sum over a bytearray. Expected result for correct firmware: 0x00D2F2FF."""
    return sum(firmware_bytes) & 0xFFFFFFFF


def build_firmware_binary(lines_dict):
    """
    Build 81,920-byte firmware from a lines dict.

    Args:
        lines_dict: {addr_hex_str: [byte_hex_str, ...]} where addr is 5-digit uppercase hex
                    and bytes are 2-digit uppercase hex strings. "--" bytes are treated as 0xFF.

    Returns:
        bytearray of length BUFFER_SIZE, pre-filled with 0xFF.
    """
    firmware = bytearray([0xFF] * BUFFER_SIZE)
    for addr_str, byte_list in lines_dict.items():
        addr = int(addr_str, 16)
        offset = addr - BASE_ADDR
        if 0 <= offset <= BUFFER_SIZE - 16:
            for i, b in enumerate(byte_list):
                if b != "--":
                    firmware[offset + i] = int(b, 16)
    return firmware


def build_rom_binary(lines_dict):
    """
    Build 65,536-byte ROM-only binary from a lines dict ($04000-$13FFF).

    Args:
        lines_dict: {addr_hex_str: [byte_hex_str, ...]} where addr is 5-digit uppercase hex

    Returns:
        bytearray of length ROM_SIZE, pre-filled with 0xFF.
    """
    firmware = build_firmware_binary(lines_dict)
    return firmware[ROM_START - BASE_ADDR:ROM_START - BASE_ADDR + ROM_SIZE]
