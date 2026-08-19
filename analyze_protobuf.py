#!/usr/bin/env python3
"""Analyze DJI protobuf metadata to reverse-engineer schema."""

import struct
import sys
from typing import List, Tuple

def parse_varint(data: bytes, offset: int) -> Tuple[int, int]:
    """Parse a varint and return (value, new_offset)."""
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        result |= (byte & 0x7f) << shift
        offset += 1
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, offset

def parse_field_header(data: bytes, offset: int) -> Tuple[int, int, int]:
    """Parse field header: returns (field_number, wire_type, new_offset)."""
    header, offset = parse_varint(data, offset)
    field_number = header >> 3
    wire_type = header & 0x7
    return field_number, wire_type, offset

def parse_message(data: bytes, name: str = "Message", max_depth: int = 3, depth: int = 0, max_fields: int = 50):
    """Parse and display protobuf message structure."""
    offset = 0
    field_count = 0
    indent = "  " * depth

    print(f"{indent}{name} (size: {len(data)} bytes)")

    while offset < len(data) and field_count < max_fields:
        if offset + 1 > len(data):
            break

        field_num, wire_type, new_offset = parse_field_header(data, offset)

        # Wire types: 0=varint, 1=64-bit, 2=length-delimited, 3=start_group, 4=end_group, 5=32-bit
        wire_type_names = ["varint", "64-bit", "length-delimited", "start_group", "end_group", "32-bit"]

        if new_offset > len(data):
            break

        print(f"{indent}  Field {field_num} ({wire_type_names[wire_type] if wire_type < 6 else '?'}):", end="")

        try:
            if wire_type == 0:  # varint
                value, new_offset = parse_varint(data, new_offset)
                print(f" {value}")

            elif wire_type == 1:  # 64-bit
                if new_offset + 8 <= len(data):
                    value = struct.unpack('<d', data[new_offset:new_offset+8])[0]
                    print(f" {value} (float64)")
                    new_offset += 8

            elif wire_type == 2:  # length-delimited (string, bytes, embedded message)
                length, new_offset = parse_varint(data, new_offset)
                if new_offset + length <= len(data):
                    field_data = data[new_offset:new_offset+length]
                    print(f" length={length}", end="")

                    # Try to detect if it's text
                    try:
                        text = field_data.decode('utf-8')
                        if all(32 <= ord(c) < 127 or c in '\n\r\t' for c in text):
                            print(f" (text): {repr(text[:50])}")
                        else:
                            print(f" (binary/message)")
                            if depth < max_depth and length > 4:
                                parse_message(field_data, f"Field{field_num}", max_depth, depth + 1, max_fields=20)
                    except:
                        print(f" (binary/message)")
                        if depth < max_depth and length > 4:
                            parse_message(field_data, f"Field{field_num}", max_depth, depth + 1, max_fields=20)

                    new_offset += length

            elif wire_type == 5:  # 32-bit
                if new_offset + 4 <= len(data):
                    value = struct.unpack('<f', data[new_offset:new_offset+4])[0]
                    print(f" {value} (float32)")
                    new_offset += 4

            else:
                print(f" (unsupported wire_type {wire_type})")

        except Exception as e:
            print(f" ERROR: {e}")
            break

        offset = new_offset
        field_count += 1

    if field_count >= max_fields:
        print(f"{indent}  ... (truncated after {max_fields} fields)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze_protobuf.py <binary_file>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        data = f.read()

    print(f"Analyzing {len(data)} bytes of protobuf data\n")
    parse_message(data, "DJI_Metadata_Stream")
