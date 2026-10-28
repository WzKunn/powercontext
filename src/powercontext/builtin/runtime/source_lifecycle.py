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

"""Track the availability of Memory and Topic Memory evidence by Source.

A scope owns one directed acyclic graph of evidence dependencies.  Sources
are the roots: withdrawing a Source withdraws the Memory entries and Topic
Memory summaries that depend on it, and restoring it restores them again.
The body of a registered entry never changes, so every citation keeps
pointing at the version that was captured.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.source_lifecycle import SourceLifecycleNode
from powercontext.sources.models import SourceRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    from powercontext.builtin.runtime.application import BuiltinRuntime

SOURCE_LIFECYCLE_SOURCE_KIND = "source"
SOURCE_LIFECYCLE_MEMORY_KIND = "memory"
SOURCE_LIFECYCLE_TOPIC_KIND = "topic"
NonEmptyText = Annotated[str, Field(min_length=1)]


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration, event or lookup breaks the contract."""


class SourceLifecycleEvent(BaseModel):
    """One availability event of one registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: NonEmptyText
    revision: Annotated[int, Field(ge=1)]
    sequence: Annotated[int, Field(ge=1)]
    available: bool
    reason: NonEmptyText

    @field_validator("node_id", "reason")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle identity and reason must not be blank")  # noqa: TRY003
        return value


def _require_node_id(node_id: str) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node_id must be a non-blank string")
    if node_id != node_id.strip():
        raise SourceLifecycleError("node_id must be trimmed")
    return node_id


def _require_policy(policy: str) -> str:
    if policy not in {"all", "any"}:
        raise SourceLifecycleError("policy must be 'all' or 'any'")
    return policy


def _require_parents(parents: Sequence[str]) -> tuple[str, ...]:
    values = tuple(parents)
    if not values:
        raise SourceLifecycleError("a derived node needs at least one parent")
    if len(set(values)) != len(values):
        raise SourceLifecycleError("a derived node must not repeat a parent")
    for parent in values:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("parent identifiers must be non-blank strings")
    return values


class ScopedSourceLifecycleApplication:
    """Register and apply the evidence dependencies of one scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        *,
        revision: int = 3,
    ) -> None:
        """Register one captured Source as the root of the evidence graph."""

        node = _require_node_id(node_id)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise SourceLifecycleError("revision must be a positive integer")
        await self._require_readable_source(source_ref)
        await self._register(
            SourceLifecycleNode(
                node_id=node,
                kind=SOURCE_LIFECYCLE_SOURCE_KIND,
                reference=source_ref.model_dump(mode="json"),
                revision=revision,
            )
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: Sequence[str] = (),
        policy: str = "all",
    ) -> None:
        """Register one Memory entry and the Sources it depends on."""

        node = _require_node_id(node_id)
        selected = _require_parents(parents)
        rule = _require_policy(policy)
        await self._require_readable_memory(citation)
        await self._register(
            SourceLifecycleNode(
                node_id=node,
                kind=SOURCE_LIFECYCLE_MEMORY_KIND,
                reference=citation.model_dump(mode="json"),
                parents=selected,
                policy=rule,
            )
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: Sequence[str] = (),
        policy: str = "all",
    ) -> None:
        """Register one Topic Memory summary and the evidence it depends on."""

        node = _require_node_id(node_id)
        selected = _require_parents(parents)
        rule = _require_policy(policy)
        await self._require_readable_topic(artifact_ref)
        await self._register(
            SourceLifecycleNode(
                node_id=node,
                kind=SOURCE_LIFECYCLE_TOPIC_KIND,
                reference=artifact_ref.model_dump(mode="json"),
                parents=selected,
                policy=rule,
            )
        )

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one availability event and return the available node ids."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires a SourceLifecycleEvent")
        nodes = await self._nodes()
        current = nodes.get(event.node_id)
        if current is None or current.kind != SOURCE_LIFECYCLE_SOURCE_KIND:
            raise SourceLifecycleError(f"Source {event.node_id!r} is not registered")
        if current.revision != event.revision:
            raise SourceLifecycleError(f"Source {event.node_id!r} has another lifecycle revision")
        if event.sequence == current.sequence:
            if current.available == event.available:
                return self._available_ids(nodes)
            raise SourceLifecycleError(f"Source {event.node_id!r} already applied another event")
        if event.sequence < current.sequence:
            raise SourceLifecycleError(f"Source {event.node_id!r} already applied a later event")
        updated = {
            **nodes,
            event.node_id: current.model_copy(
                update={"available": event.available, "sequence": event.sequence}
            ),
        }
        updated = self._propagate(updated, event.node_id)
        await self._apply_native_effects(updated, nodes)
        for node_id, node in updated.items():
            before = nodes.get(node_id)
            if before is None or before.available != node.available or before.sequence != node.sequence:
                await self._save(node)
        return self._available_ids(updated)

    async def resolve(self, node_id: str, *, purpose: str = "generation") -> Any:
        """Return the exact reference of one node for the requested purpose."""

        node = _require_node_id(node_id)
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose must be 'generation' or 'audit'")
        nodes = await self._nodes()
        current = nodes.get(node)
        if current is None:
            raise SourceLifecycleError(f"node {node!r} is not registered")
        if purpose == "generation" and not current.available:
            raise SourceLifecycleError(f"node {node!r} is not available for generation")
        return _decode_reference(current)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    async def _register(self, node: SourceLifecycleNode) -> None:
        nodes = await self._nodes()
        existing = nodes.get(node.node_id)
        if existing is not None:
            if _same_binding(existing, node):
                return
            raise SourceLifecycleError(f"node {node.node_id!r} is already bound to other evidence")
        for parent in node.parents:
            if parent not in nodes:
                raise SourceLifecycleError(f"parent {parent!r} is not registered")
        if node.node_id in node.parents:
            raise SourceLifecycleError("a node must not depend on itself")
        self._reject_cycle(nodes, node)
        available = self._availability(nodes, node)
        await self._save(node.model_copy(update={"available": available}))
        if node.kind == SOURCE_LIFECYCLE_MEMORY_KIND:
            await self._set_memory_state(node, available)

    def _reject_cycle(self, nodes: dict[str, SourceLifecycleNode], node: SourceLifecycleNode) -> None:
        pending = list(node.parents)
        visited: set[str] = set()
        while pending:
            parent = pending.pop()
            if parent == node.node_id:
                raise SourceLifecycleError("the dependency graph must not contain a cycle")
            if parent in visited:
                continue
            visited.add(parent)
            known = nodes.get(parent)
            if known is not None:
                pending.extend(known.parents)

    def _availability(self, nodes: dict[str, SourceLifecycleNode], node: SourceLifecycleNode) -> bool:
        """Compute the availability of one derived node from its parents."""

        if not node.parents:
            return True
        support = [nodes[parent].available for parent in node.parents if parent in nodes]
        if len(support) != len(node.parents):
            raise SourceLifecycleError("every parent must be registered")
        if node.policy == "all":
            return all(support)
        return any(support)

    def _propagate(
        self,
        nodes: dict[str, SourceLifecycleNode],
        changed: str,
    ) -> dict[str, SourceLifecycleNode]:
        """Propagate one availability change to every descendant."""

        children: dict[str, list[str]] = {}
        for node in nodes.values():
            for parent in node.parents:
                children.setdefault(parent, []).append(node.node_id)
        pending = list(children.get(changed, ()))
        visited: set[str] = set()
        while pending:
            node_id = pending.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            node = nodes[node_id]
            availability = self._availability(nodes, node)
            if availability != node.available:
                nodes[node_id] = node.model_copy(update={"available": availability})
                pending.extend(children.get(node_id, ()))
        return nodes

    @staticmethod
    def _available_ids(nodes: dict[str, SourceLifecycleNode]) -> list[str]:
        return sorted(node_id for node_id, node in nodes.items() if node.available)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    async def _nodes(self) -> dict[str, SourceLifecycleNode]:
        rows = await self._runtime.source_lifecycle.nodes(self.scope_id)
        return {row.node_id: row for row in rows}

    async def _save(self, node: SourceLifecycleNode) -> None:
        await self._runtime.source_lifecycle.save(self.scope_id, node)

    # ------------------------------------------------------------------
    # native effects
    # ------------------------------------------------------------------

    async def _apply_native_effects(
        self,
        nodes: dict[str, SourceLifecycleNode],
        previous: dict[str, SourceLifecycleNode],
    ) -> None:
        for node_id, node in nodes.items():
            before = previous.get(node_id)
            if before is not None and before.available == node.available:
                continue
            if node.kind != SOURCE_LIFECYCLE_MEMORY_KIND:
                continue
            await self._set_memory_state(node, node.available)

    async def _set_memory_state(self, node: SourceLifecycleNode, available: bool) -> None:
        """Move one Memory entry between the active and inactive states."""

        from powercontext.errors import ArtifactNotFoundError

        citation = MemoryCitation.model_validate(node.reference)
        async with self._runtime._context(self.scope_id) as context:
            async with self._runtime._locked(self.scope_id):
                service = context.artifacts.memory
                try:
                    current = await service.head(context.artifacts.memory_artifact_id)
                except ArtifactNotFoundError:
                    return
                entry = None
                for candidate in await service.entries(current):
                    if candidate.entry_id != citation.entry_id:
                        continue
                    entry = candidate
                    break
                if entry is None:
                    return
                if available:
                    await service.reactivate(current, entries=(entry,))
                else:
                    await service.forget(current, entries=(entry,))

    # ------------------------------------------------------------------
    # readability
    # ------------------------------------------------------------------

    async def _require_readable_source(self, source_ref: SourceRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                source_ref.source_type,
                source_ref.source_id,
            )
        except Exception as error:  # noqa: BLE001 - every unreadable Source is a contract error
            raise SourceLifecycleError(f"Source {source_ref.source_id!r} is not readable") from error

    async def _require_readable_memory(self, citation: MemoryCitation) -> None:
        from powercontext.builtin.runtime.models import GetMemoryEntryRequest

        try:
            await self._runtime.memory.for_scope(self.scope_id).get(
                GetMemoryEntryRequest(citation=citation)
            )
        except Exception as error:  # noqa: BLE001 - every unreadable entry is a contract error
            raise SourceLifecycleError(f"Memory entry {citation.entry_id!r} is not readable") from error

    async def _require_readable_topic(self, artifact_ref: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                artifact_ref.family,
                artifact_ref.artifact_id,
                artifact_ref.revision,
            )
        except Exception as error:  # noqa: BLE001 - every unreadable Topic is a contract error
            raise SourceLifecycleError(f"Topic {artifact_ref.artifact_id!r} is not readable") from error


class SourceLifecycleApplication:
    """Select the lifecycle application service for one scope."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id)


def _same_binding(left: SourceLifecycleNode, right: SourceLifecycleNode) -> bool:
    """Return whether two registrations describe the same evidence binding."""

    return (
        left.kind == right.kind
        and left.reference == right.reference
        and left.parents == right.parents
        and left.policy == right.policy
        and left.revision == right.revision
    )


def _decode_reference(node: SourceLifecycleNode) -> Any:
    """Decode the stored reference of one node into its public value."""

    if node.kind == SOURCE_LIFECYCLE_SOURCE_KIND:
        return SourceRef.model_validate(node.reference)
    if node.kind == SOURCE_LIFECYCLE_MEMORY_KIND:
        return MemoryCitation.model_validate(node.reference)
    return ArtifactRef.model_validate(node.reference)


__all__ = [
    "SOURCE_LIFECYCLE_MEMORY_KIND",
    "SOURCE_LIFECYCLE_SOURCE_KIND",
    "SOURCE_LIFECYCLE_TOPIC_KIND",
    "ScopedSourceLifecycleApplication",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleNode",
]
