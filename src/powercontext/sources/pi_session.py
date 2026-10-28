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

"""Capture the visible events of one Pi session branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
NonEmptyText = Annotated[str, Field(min_length=1)]


class PiSessionInput(BaseModel):
    """One Pi session archive and the branch it should be read from."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event of a Pi session branch."""

    session_id: NonEmptyText
    event_id: NonEmptyText
    parent_id: str | None = None
    event_type: NonEmptyText
    role: NonEmptyText
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """Captured Pi event stored with the identity of its session node."""

    source_id: NonEmptyText
    session_id: NonEmptyText
    event_id: NonEmptyText
    parent_id: str | None = None
    event_type: NonEmptyText
    role: NonEmptyText
    content: str
    record: dict[str, Any]


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    """Return the stable Source identity of one Pi session event."""

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")  # noqa: TRY003
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse the visible events of the selected branch of one Pi session.

    The whole archive is parsed and validated before any event is returned,
    so a broken archive fails atomically instead of yielding a partial
    branch.
    """

    records = _parse_records(value.content)
    header = records[0]
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must carry a non-empty id")  # noqa: TRY003
    nodes: dict[str, dict[str, Any]] = {}
    parents: dict[str, str | None] = {}
    for record in records[1:]:
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Pi session nodes must carry a non-empty id")  # noqa: TRY003
        if node_id in nodes:
            raise ValueError(f"Pi session node {node_id!r} is duplicated")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("Pi session parentId must be a string or null")  # noqa: TRY003
        if parent_id and parent_id not in nodes:
            raise ValueError(f"Pi session node {node_id!r} has an unknown parent")  # noqa: TRY003
        nodes[node_id] = record
        parents[node_id] = parent_id or None
    leaf_id = value.leaf_id or records[-1].get("id")
    if not isinstance(leaf_id, str) or leaf_id not in nodes:
        raise ValueError("Pi session leaf is not part of the archive")  # noqa: TRY003
    chain: list[str] = []
    cursor: str | None = leaf_id
    seen: set[str] = set()
    while cursor is not None:
        if cursor in seen:
            raise ValueError("Pi session branch contains a cycle")  # noqa: TRY003
        seen.add(cursor)
        chain.append(cursor)
        cursor = parents[cursor]
    events: list[PiEventInput] = []
    for node_id in reversed(chain):
        record = nodes[node_id]
        visible = _visible_event(record)
        if visible is None:
            continue
        role, body = visible
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=node_id,
                parent_id=parents[node_id],
                event_type=str(record.get("type") or "message"),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read the Pi event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured Pi event as its complete visible text."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiSessionConnector:
    """Capture the visible events of one Pi session branch, incrementally."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None, /) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Submit the events of the selected branch that are not captured yet."""

        raw = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))
        captured = _captured_events(session.checkpoint)
        for event in events:
            digest = _event_digest(event)
            previous = captured.get(event.event_id)
            if previous is not None:
                if previous != digest:
                    raise ValueError(f"Pi session event {event.event_id!r} changed after capture")  # noqa: TRY003
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            captured[event.event_id] = digest
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"events": dict(sorted(captured.items()))},
        )


def _captured_events(checkpoint: object) -> dict[str, str]:
    """Read the captured event digests of one persisted checkpoint."""

    if not isinstance(checkpoint, dict):
        return {}
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in events.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _event_digest(event: PiEventInput, /) -> str:
    """Return the digest of one captured event payload."""

    encoded = json.dumps(
        {"content": event.content, "record": event.record},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_records(content: bytes, /) -> list[dict[str, Any]]:
    """Parse one JSONL session archive into its validated records."""

    if not content:
        raise ValueError("Pi session archive must not be empty")  # noqa: TRY003
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ValueError(f"Pi session line {number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {number} must be a JSON object")  # noqa: TRY003
        records.append(record)
    if not records:
        raise ValueError("Pi session archive must contain a session header")  # noqa: TRY003
    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session header must be a version 3 session record")  # noqa: TRY003
    return records


def _visible_event(record: dict[str, Any], /) -> tuple[str, str] | None:
    """Return the visible role and body of one record, or None for control nodes."""

    event_type = record.get("type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        if role == "bashExecution":
            command = message.get("command")
            output = message.get("output")
            return role, f"$ {command if isinstance(command, str) else ''}\n{output if isinstance(output, str) else ''}"
        if role == "toolResult":
            tool_name = message.get("toolName")
            call_id = message.get("toolCallId")
            body = _message_body(message.get("content"))
            return role, (
                f"[tool_result {tool_name if isinstance(tool_name, str) else ''} "
                f"{call_id if isinstance(call_id, str) else ''}] {body or ''}"
            )
        body = _message_body(message.get("content"))
        if body is None:
            return None
        return role, body
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        return str(event_type), summary if isinstance(summary, str) else ""
    if event_type == "custom_message":
        body = _message_body(record.get("content"))
        if body is None:
            return None
        return "custom", body
    return None


def _message_body(content: object, /) -> str | None:
    """Render the visible text of one message payload."""

    if isinstance(content, str):
        return content if content.strip() else None
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            encoded = json.dumps(
                arguments if arguments is not None else {},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            parts.append(
                f"[tool_call {name if isinstance(name, str) else ''} {call_id if isinstance(call_id, str) else ''}] {encoded}"
            )
    if not parts:
        return None
    return "\n".join(parts)


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)

__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
