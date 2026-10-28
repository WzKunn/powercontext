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

"""Persist the Source lifecycle graph of one scope."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.persistence.database import AsyncDatabase

NonEmptyText = Annotated[str, Field(min_length=1)]


class SourceLifecycleNode(BaseModel):
    """One registered node of the evidence graph of one scope."""

    model_config = ConfigDict(frozen=True)

    node_id: NonEmptyText
    kind: Literal["source", "memory", "topic"]
    reference: dict[str, Any]
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int = 1
    sequence: int = 0
    available: bool = True


class RelationalSourceLifecycle:
    """Store and read the Source lifecycle graph of every scope."""

    def __init__(self, database: AsyncDatabase) -> None:
        self._database = database

    async def nodes(self, scope_id: str, /) -> tuple[SourceLifecycleNode, ...]:
        """Return every registered node of one scope."""

        async with self._database.transaction() as connection:
            statement = select(SOURCE_LIFECYCLE_NODES_TABLE).where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id
            )
            rows = (await connection.execute(statement)).mappings().all()
        return tuple(_decode(row) for row in rows)

    async def save(self, scope_id: str, node: SourceLifecycleNode, /) -> None:
        """Persist one node, replacing the previous registration of the same id."""

        table = SOURCE_LIFECYCLE_NODES_TABLE
        async with self._database.transaction() as connection:
            await connection.execute(
                delete(table).where(table.c.scope_id == scope_id, table.c.node_id == node.node_id)
            )
            await connection.execute(
                insert(table).values(
                    scope_id=scope_id,
                    node_id=node.node_id,
                    kind=node.kind,
                    reference=json.dumps(node.reference, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    parents=json.dumps(list(node.parents), ensure_ascii=False, separators=(",", ":")),
                    policy=node.policy,
                    revision=node.revision,
                    sequence=node.sequence,
                    available=node.available,
                )
            )

    async def unavailable_topic_ids(self, connection: AsyncConnection, scope_id: str, /) -> frozenset[str]:
        """Return the Topic artifacts that no longer have usable support."""

        table = SOURCE_LIFECYCLE_NODES_TABLE
        statement = select(table.c.reference).where(
            table.c.scope_id == scope_id,
            table.c.kind == "topic",
            table.c.available.is_(False),
        )
        rows = (await connection.execute(statement)).all()
        artifacts: set[str] = set()
        for row in rows:
            reference = json.loads(row[0])
            artifact_id = reference.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                artifacts.add(artifact_id)
        return frozenset(artifacts)


def _decode(row: Any) -> SourceLifecycleNode:
    """Decode one stored row into its node value."""

    return SourceLifecycleNode(
        node_id=str(row["node_id"]),
        kind=str(row["kind"]),
        reference=json.loads(row["reference"]),
        parents=tuple(json.loads(row["parents"])),
        policy=str(row["policy"]),
        revision=int(row["revision"]),
        sequence=int(row["sequence"]),
        available=bool(row["available"]),
    )


__all__ = ["RelationalSourceLifecycle", "SourceLifecycleNode"]
