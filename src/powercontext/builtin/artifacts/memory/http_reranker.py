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

"""Rerank coarse Memory candidates with an external HTTP scoring service."""

from __future__ import annotations

import math

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
DEFAULT_HTTP_RERANK_BATCH_SIZE = 32


class _ScoreRecord(BaseModel):
    """One service score bound to the global rank of its candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class _ScoreResponse(BaseModel):
    """The exact response shape accepted from the scoring service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: tuple[_ScoreRecord, ...]


class HTTPMemoryReranker:
    """Score coarse Memory candidates with a batched HTTP scoring service."""

    policy_id = HTTP_RERANK_POLICY_ID

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_HTTP_RERANK_BATCH_SIZE,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("endpoint must be a non-empty string")  # noqa: TRY003
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Select the globally best candidates using the service scores."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for batch in _batches(candidates, self._batch_size):
            requests += 1
            batch_scores = await self._score_batch(query, batch)
            scores.update(batch_scores)
        selected = _select_ranks(scores, limit=limit)
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=len(candidates) - len(selected),
        )

    async def _score_batch(self, query: str, batch: tuple[tuple[int, str], ...]) -> dict[int, float]:
        """Score one bounded batch and validate that it covers the batch exactly."""

        payload = {
            "query": query,
            "candidates": [{"rank": rank, "text": text} for rank, text in batch],
        }
        response = await self._client.post(self._endpoint, json=payload)
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError as error:
            raise ValueError("rerank service returned a non-JSON response") from error
        if not isinstance(body, dict):
            raise ValueError("rerank service response must be a JSON object")  # noqa: TRY003
        try:
            parsed = _ScoreResponse.model_validate(body)
        except ValidationError as error:
            raise ValueError(f"rerank service response is invalid: {error}") from error
        expected = {rank for rank, _text in batch}
        observed: set[int] = set()
        scores: dict[int, float] = {}
        for record in parsed.scores:
            if record.rank not in expected:
                raise ValueError(f"rerank service scored unknown rank {record.rank}")  # noqa: TRY003
            if record.rank in observed:
                raise ValueError(f"rerank service scored rank {record.rank} twice")  # noqa: TRY003
            if not math.isfinite(record.score):
                raise ValueError("rerank service returned a non-finite score")  # noqa: TRY003
            observed.add(record.rank)
            scores[record.rank] = record.score
        if observed != expected:
            missing = sorted(expected - observed)
            raise ValueError(f"rerank service did not score ranks {missing}")  # noqa: TRY003
        return scores


def _batches(candidates: tuple[MemoryHit, ...], batch_size: int) -> list[tuple[tuple[int, str], ...]]:
    """Split the candidate pool into bounded batches of global ranks."""

    items = tuple((rank, candidate.text) for rank, candidate in enumerate(candidates, start=1))
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _select_ranks(scores: dict[int, float], *, limit: int) -> tuple[int, ...]:
    """Choose the globally best ranks, breaking ties by the original rank."""

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(rank for rank, _score in ordered[:limit])


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["DEFAULT_HTTP_RERANK_BATCH_SIZE", "HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
