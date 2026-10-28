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

"""Merge concurrent embedding requests and cache the vectors they produce."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024


@dataclass(slots=True)
class _SharedText:
    """One text whose vector is computed once for every waiting caller."""

    text: str
    future: asyncio.Future[EmbeddingVector]
    batch_id: int | None = None


@dataclass(slots=True)
class _BatchUsage:
    """The provider usage of one underlying embedding request."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class _Coordinator:
    """The shared state of the batching coordinator of one wrapper."""

    entries: "OrderedDict[str, _SharedText]" = field(default_factory=OrderedDict)
    waiting: list[str] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    batches: dict[int, _BatchUsage] = field(default_factory=dict)
    next_batch_id: int = 0


class CoalescingEmbeddingModel:
    """Share one embedding workload across concurrent callers and cache results."""

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(coalesce_ms, (int, float)) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_ms = float(coalesce_ms)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._coordinator = _Coordinator()

    @property
    def profile(self) -> Any:
        """Expose the profile of the wrapped model."""

        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one vector per input text, sharing the underlying requests."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        vectors: list[EmbeddingVector | None] = [None] * len(texts)
        awaited: list[tuple[int, _SharedText]] = []
        for index, text in enumerate(texts):
            cached = self._cached_vector(text)
            if cached is not None:
                vectors[index] = cached
                continue
            shared = self._share(text)
            awaited.append((index, shared))
        if awaited:
            results = await asyncio.gather(
                *(asyncio.shield(shared.future) for _index, shared in awaited)
            )
            for (index, _shared), vector in zip(awaited, results, strict=True):
                vectors[index] = vector
        resolved = tuple(vector for vector in vectors if vector is not None)
        if len(resolved) != len(texts):
            raise ValueError("embedding model did not return one vector per text")  # noqa: TRY003
        return EmbeddingResult(vectors=resolved, usage=self._usage([shared for _index, shared in awaited]))

    def _cached_vector(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0 or text not in self._cache:
            return None
        vector = self._cache.pop(text)
        self._cache[text] = vector
        return vector

    def _share(self, text: str) -> _SharedText:
        """Return the shared computation of one text, starting it when needed."""

        coordinator = self._coordinator
        existing = coordinator.entries.get(text)
        if existing is not None:
            return existing
        shared = _SharedText(text=text, future=asyncio.get_running_loop().create_future())
        coordinator.entries[text] = shared
        coordinator.waiting.append(text)
        if coordinator.task is None or coordinator.task.done():
            coordinator.task = asyncio.create_task(self._coordinate())
        return shared

    async def _coordinate(self) -> None:
        """Group shared texts into bounded underlying embedding requests."""

        coordinator = self._coordinator
        while coordinator.waiting:
            if self._coalesce_ms > 0:
                await asyncio.sleep(self._coalesce_ms / 1000)
            batch = self._take_batch()
            if not batch:
                continue
            await self._run_batch(batch)

    def _take_batch(self) -> tuple[_SharedText, ...]:
        coordinator = self._coordinator
        batch: list[_SharedText] = []
        while coordinator.waiting and len(batch) < self._max_batch_size:
            text = coordinator.waiting.pop(0)
            shared = coordinator.entries.get(text)
            if shared is not None:
                batch.append(shared)
        return tuple(batch)

    async def _run_batch(self, batch: tuple[_SharedText, ...]) -> None:
        coordinator = self._coordinator
        batch_id = coordinator.next_batch_id
        coordinator.next_batch_id += 1
        for shared in batch:
            shared.batch_id = batch_id
        texts = tuple(shared.text for shared in batch)
        try:
            try:
                result = await asyncio.shield(self._model.embed(texts))
            except asyncio.CancelledError:
                for shared in batch:
                    if not shared.future.done():
                        shared.future.cancel()
                raise
            except BaseException as error:  # noqa: BLE001 - every waiter observes the provider error
                for shared in batch:
                    if not shared.future.done():
                        shared.future.set_exception(error)
                return
            vectors = tuple(result.vectors)
            if len(vectors) != len(texts):
                error = ValueError("embedding model returned a different number of vectors than texts")
                for shared in batch:
                    if not shared.future.done():
                        shared.future.set_exception(error)
                return
            coordinator.batches[batch_id] = _BatchUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            )
            for shared, vector in zip(batch, vectors, strict=True):
                self._remember(shared.text, vector)
                if not shared.future.done():
                    shared.future.set_result(vector)
        finally:
            for shared in batch:
                coordinator.entries.pop(shared.text, None)

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        self._cache.pop(text, None)
        self._cache[text] = vector
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _usage(self, awaited: list[_SharedText]) -> InferenceUsage:
        batches: dict[int, _BatchUsage] = {}
        for shared in awaited:
            if shared.batch_id is None:
                continue
            usage = self._coordinator.batches.get(shared.batch_id)
            if usage is not None:
                batches[shared.batch_id] = usage
        if not batches:
            return InferenceUsage(requests=0)
        return InferenceUsage(
            requests=len(batches),
            input_tokens=_sum_tokens([usage.input_tokens for usage in batches.values()]),
            output_tokens=_sum_tokens([usage.output_tokens for usage in batches.values()]),
        )


def _sum_tokens(values: list[int | None]) -> int | None:
    if all(value is None for value in values):
        return None
    return sum(value or 0 for value in values)


__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_COALESCE_MS",
    "DEFAULT_MAX_BATCH_SIZE",
    "CoalescingEmbeddingModel",
]
