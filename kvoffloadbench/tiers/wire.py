"""Binary wire protocol for the NVMe-oF-style TCP target.

Requests are ``!BII`` (opcode, block-id length, payload length) followed by the
UTF-8 block id and the payload. Responses are ``!BI`` (status, payload length)
followed by the payload. Frames are length-prefixed, which lets a client
pipeline many commands before reading any response.
"""

from __future__ import annotations

import struct

REQUEST_HEADER = struct.Struct("!BII")
RESPONSE_HEADER = struct.Struct("!BI")

OP_PUT = 1
OP_GET = 2
OP_CONTAINS = 3
OP_EVICT = 4
OP_STATS = 5
OP_PING = 6

STATUS_OK = 0
STATUS_NOT_FOUND = 1
STATUS_ERROR = 2

MAX_BLOCK_ID = 4096
MAX_PAYLOAD = 256 * 1024 * 1024

OP_NAMES = {
    OP_PUT: "PUT",
    OP_GET: "GET",
    OP_CONTAINS: "CONTAINS",
    OP_EVICT: "EVICT",
    OP_STATS: "STATS",
    OP_PING: "PING",
}


def encode_request(op: int, block_id: str = "", payload: bytes = b"") -> bytes:
    id_bytes = block_id.encode("utf-8")
    if len(id_bytes) > MAX_BLOCK_ID:
        raise ValueError(f"block id is too long ({len(id_bytes)} > {MAX_BLOCK_ID})")
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload is too large ({len(payload)} > {MAX_PAYLOAD})")
    return REQUEST_HEADER.pack(op, len(id_bytes), len(payload)) + id_bytes + payload


def encode_response(status: int, payload: bytes = b"") -> bytes:
    return RESPONSE_HEADER.pack(status, len(payload)) + payload
