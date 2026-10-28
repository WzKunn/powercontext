# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunked Context Packs and the incremental deltas that rebuild them."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CONTEXT_CHUNK_BYTES = 512
RUNTIME_CONTEXT_CHUNK_BYTES = 128
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when a Context Pack or delta violates the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete, verifiable package of prepared context bytes."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...], /) -> ContextPack:
        """Build a complete pack from exact byte items."""

        values = _validate_items(items)
        digest = _pack_digest(values)
        byte_count = _resolve_byte_count(values, digest)
        return cls(items=values, digest=digest, byte_count=byte_count)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CONTEXT_CHUNK_BYTES) -> ContextPack:
        """Chunk prepared context deterministically without changing its bytes."""

        if not isinstance(content, str):
            raise ContextPackError("content must be a string")
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        return cls.build(_chunk_bytes(content.encode("utf-8"), chunk_bytes))

    def to_bytes(self) -> bytes:
        """Serialize the pack into its canonical wire representation."""

        return _serialize_pack(self.items, self.digest, self.byte_count)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and verify one complete pack."""

        payload = _decode_object(data, "Context Pack")
        _require_fields(payload, _PACK_FIELDS, "Context Pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("Context Pack schema is not supported")
        items = _decode_items(payload["items"], "Context Pack")
        digest = _require_digest(payload["digest"], "Context Pack")
        byte_count = _require_integer(payload["byte_count"], "Context Pack")
        expected = _pack_digest(items)
        if digest != expected:
            raise ContextPackError("Context Pack digest does not match its items")
        if byte_count != len(data):
            raise ContextPackError("Context Pack byte count does not match its payload")
        return cls(items=items, digest=digest, byte_count=byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """One incremental update that rebuilds a target pack from a base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        """Describe how the target pack extends the base pack."""

        prefix_items = 0
        for base_item, target_item in zip(base.items, target.items, strict=False):
            if base_item != target_item:
                break
            prefix_items += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix_items,
            appended=target.items[prefix_items:],
        )

    def to_bytes(self) -> bytes:
        """Serialize the delta into its canonical wire representation."""

        payload = {
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_encode_item(item) for item in self.appended],
        }
        return _canonical_json(payload)

    @property
    def byte_count(self) -> int:
        """Return the exact serialized size of this delta."""

        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and verify one delta."""

        payload = _decode_object(data, "Context Pack delta")
        _require_fields(payload, _DELTA_FIELDS, "Context Pack delta")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("Context Pack delta schema is not supported")
        base_digest = _require_digest(payload["base_digest"], "Context Pack delta")
        target_digest = _require_digest(payload["target_digest"], "Context Pack delta")
        prefix_items = _require_prefix(payload["prefix_items"])
        appended = _decode_items(payload["appended"], "Context Pack delta")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a base pack and one delta."""

    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a Context Pack")
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta must be a Context Pack delta")
    if base.digest != delta.base_digest:
        raise ContextPackError("Context Pack delta does not match the base pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("Context Pack delta prefix exceeds the base pack")
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("Context Pack delta does not rebuild its target pack")
    return target


def incremental_context_pack(
    base: ContextPack | None,
    content: str,
    /,
    *,
    chunk_bytes: int = RUNTIME_CONTEXT_CHUNK_BYTES,
) -> ContextPack | ContextPackDelta:
    """Return the complete pack for ``content`` or the delta from ``base``.

    The delta reuses every byte the base pack already carries, so only the
    changed tail travels on the wire.  The rebuilt pack contains exactly the
    same bytes as the complete pack of the current content.
    """

    if base is None:
        return ContextPack.from_content(content, chunk_bytes=chunk_bytes)
    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a Context Pack")
    target_bytes = content.encode("utf-8") if isinstance(content, str) else b""
    base_bytes = b"".join(base.items)
    common = _common_prefix_length(base_bytes, target_bytes)
    reused: list[bytes] = []
    consumed = 0
    for item in base.items:
        if consumed + len(item) > common:
            break
        consumed += len(item)
        reused.append(item)
    target = ContextPack.build((*reused, *_chunk_bytes(target_bytes[consumed:], chunk_bytes)))
    return ContextPackDelta.between(base, target)


def _common_prefix_length(left: bytes, right: bytes, /) -> int:
    """Return the length of the shared prefix of two byte strings."""

    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _pack_digest(items: tuple[bytes, ...], /) -> str:
    payload = {"schema": CONTEXT_PACK_SCHEMA, "items": [_encode_item(item) for item in items]}
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _resolve_byte_count(items: tuple[bytes, ...], digest: str, /) -> int:
    guess = 0
    for _attempt in range(8):
        data = _serialize_pack(items, digest, guess)
        if len(data) == guess:
            return guess
        guess = len(data)
    raise ContextPackError("Context Pack byte count cannot be resolved")  # noqa: TRY003


def _serialize_pack(items: tuple[bytes, ...], digest: str, byte_count: int, /) -> bytes:
    payload = {
        "schema": CONTEXT_PACK_SCHEMA,
        "items": [_encode_item(item) for item in items],
        "digest": digest,
        "byte_count": byte_count,
    }
    return _canonical_json(payload)


def _canonical_json(payload: dict[str, Any], /) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_item(item: bytes, /) -> str:
    return base64.b64encode(item).decode("ascii")


def _validate_items(items: object, /) -> tuple[bytes, ...]:
    if isinstance(items, (bytes, bytearray, str)) or not isinstance(items, (tuple, list)):
        raise ContextPackError("items must be a sequence of byte strings")
    values: list[bytes] = []
    for item in items:
        if not isinstance(item, bytes):
            raise ContextPackError("items must contain only byte strings")
        values.append(item)
    return tuple(values)


def _decode_object(data: bytes, label: str, /) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)):
        raise ContextPackError(f"{label} payload must be bytes")
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContextPackError(f"{label} payload is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} payload must be a JSON object")
    return payload


def _require_fields(payload: dict[str, Any], expected: frozenset[str], label: str, /) -> None:
    if set(payload) != expected:
        raise ContextPackError(f"{label} payload must contain exactly the fields {sorted(expected)}")


def _decode_items(value: object, label: str, /) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{label} items must be a list of Base64 strings")
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{label} items must be Base64 strings")
        try:
            items.append(base64.b64decode(entry.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError, ValueError) as error:
            raise ContextPackError(f"{label} items must be valid Base64") from error
    return tuple(items)


def _require_digest(value: object, label: str, /) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContextPackError(f"{label} digest must use sha256:<hex>")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ContextPackError(f"{label} digest must contain hexadecimal characters") from error
    return value


def _require_integer(value: object, label: str, /) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContextPackError(f"{label} byte count must be an integer")
    return value


def _require_prefix(value: object, /) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContextPackError("Context Pack delta prefix must be an integer")
    if value < 0:
        raise ContextPackError("Context Pack delta prefix must not be negative")
    return value


def _chunk_bytes(data: bytes, chunk_bytes: int, /) -> tuple[bytes, ...]:
    """Split content into deterministic chunks that keep every line intact."""

    items: list[bytes] = []
    current = bytearray()
    for line in data.splitlines(keepends=True):
        if len(line) > chunk_bytes:
            if current:
                items.append(bytes(current))
                current.clear()
            items.extend(_split_utf8_line(line, chunk_bytes))
            continue
        if len(current) + len(line) > chunk_bytes:
            items.append(bytes(current))
            current.clear()
        current.extend(line)
    if current:
        items.append(bytes(current))
    return tuple(items)


def _split_utf8_line(line: bytes, chunk_bytes: int, /) -> list[bytes]:
    """Split one long line into valid UTF-8 pieces within the byte limit."""

    pieces: list[bytes] = []
    current = bytearray()
    for character in line.decode("utf-8"):
        encoded = character.encode("utf-8")
        if len(encoded) > chunk_bytes:
            raise ContextPackError("chunk_bytes is too small for the content")
        if len(current) + len(encoded) > chunk_bytes:
            pieces.append(bytes(current))
            current.clear()
        current.extend(encoded)
    if current:
        pieces.append(bytes(current))
    return pieces


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CONTEXT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
    "incremental_context_pack",
]
