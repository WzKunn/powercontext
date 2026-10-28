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

"""Page-level PDF capture with a stable identity and exact page evidence."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Protocol
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
PDF_PAGE_IDENTITY_PREFIX = "pdf"
NonEmptyText = Annotated[str, Field(min_length=1)]


class PDFDocumentInput(BaseModel):
    """One complete PDF document captured page by page."""

    document_id: NonEmptyText
    content: bytes

    @field_validator("document_id")
    @classmethod
    def reject_blank_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """One physical page of a PDF document with its document identity."""

    document_id: NonEmptyText
    document_sha256: NonEmptyText
    page_number: Annotated[int, Field(ge=1)]
    text: str


class PDFPageSource(Source):
    """Captured PDF page stored with the identity of its physical page."""

    source_id: NonEmptyText
    document_id: NonEmptyText
    document_sha256: NonEmptyText
    page_number: Annotated[int, Field(ge=1)]
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int, /) -> str:
    """Return the stable Source identity of one physical PDF page."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")  # noqa: TRY003
    if not isinstance(document_sha256, str) or not document_sha256:
        raise ValueError("document_sha256 must be a non-empty string")  # noqa: TRY003
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive integer")  # noqa: TRY003
    identity = (
        f"{PDF_PAGE_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    )
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF document.

    The whole document is parsed before any page is returned, so a broken
    document fails atomically instead of yielding a partial page list.
    """

    from pypdf import PdfReader

    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    digest = hashlib.sha256(content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF document requires a password")  # noqa: TRY003
        pages = list(reader.pages)
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - every unreadable document is a ValueError
        raise ValueError(f"PDF document could not be parsed: {error}") from error
    if not pages:
        raise ValueError("PDF document must contain at least one page")  # noqa: TRY003
    parsed: list[PDFPageInput] = []
    for number, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:  # noqa: BLE001 - every unreadable page fails the document
            raise ValueError(f"PDF page {number} could not be parsed: {error}") from error
        parsed.append(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=number,
                text=text,
            )
        )
    return tuple(parsed)


class PDFPageSourceAdapter:
    """Resolve and read the page-level PDF Source."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one captured PDF page as textual evidence of that page."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _ScopedMemory(Protocol):
    """The scoped Memory service used to search the captured pages."""

    async def search(self, request: SearchMemoryRequest, /) -> Any: ...


_PDF_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "give",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)
_PDF_QUERY_TERM = re.compile(r"[a-z0-9]+")


def _query_terms(question: str) -> tuple[str, ...]:
    """Return the informative lowercase terms of one question."""

    terms = (token for token in _PDF_QUERY_TERM.findall(question.casefold()))
    return tuple(dict.fromkeys(token for token in terms if len(token) > 1 and token not in _PDF_QUERY_STOPWORDS))


def _term_overlap(text: str, terms: tuple[str, ...]) -> float:
    """Score how many distinct question terms one page actually states."""

    if not terms:
        return 0.0
    folded = text.casefold()
    matched = 0.0
    for term in terms:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded):
            matched += 1.0
    return matched / len(terms)


def _term_window(text: str, terms: tuple[str, ...]) -> int:
    """Return the smallest number of words that carries every matched term.

    An evidence page usually states the question terms close to each other, so
    a tight window separates the page that answers the question from a page
    that merely repeats one common term somewhere else.
    """

    if not terms:
        return 0
    words = _PDF_QUERY_TERM.findall(text.casefold())
    positions: dict[str, list[int]] = {}
    for index, word in enumerate(words):
        positions.setdefault(word, []).append(index)
    matched = [positions[term] for term in terms if positions.get(term)]
    if not matched:
        return 0
    best = len(words)
    for start in matched[0]:
        window = start
        for group in matched[1:]:
            window = min(window, min((item for item in group if item >= window), default=window))
        end = max((max(group) for group in matched), default=start)
        best = min(best, max(1, end - window + 1))
    return best


def _page_score(text: str, terms: tuple[str, ...], weights: Mapping[str, float]) -> float:
    """Score one page by the rarity weighted question terms it states."""

    if not terms:
        return 0.0
    folded = text.casefold()
    coverage = 0.0
    matched = 0
    for term, weight in weights.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded):
            coverage += weight
            matched += 1
    if not matched:
        return 0.0
    window = _term_window(text, terms)
    tightness = 1.0 / (1.0 + math.log1p(window)) if window else 0.0
    return coverage * (1.0 + tightness)


def _term_weights(terms: tuple[str, ...], corpus: Sequence[str]) -> dict[str, float]:
    """Weight every question term by how rare it is in the candidate pages."""

    if not terms:
        return {}
    frequency: Counter[str] = Counter()
    size = max(1, len(corpus))
    for text in corpus:
        folded = text.casefold()
        for term in terms:
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded):
                frequency[term] += 1
    return {term: math.log(1 + size / (1 + frequency[term])) for term in terms}


async def search_pdf_pages(
    memory: _ScopedMemory,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return the exact citations of the pages that answer one question.

    The search runs through the public Memory API, so every returned
    citation can be read back with :meth:`memory.get` and still names the
    physical page it was captured from.  The coarse fulltext candidates are
    re-ranked by the question terms they state, because a page that mentions
    every term of the question is a better evidence page than the first page
    that merely matches a common term.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace content")  # noqa: TRY003
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003
    terms = _query_terms(question)
    page = await memory.search(
        SearchMemoryRequest(query=question, mode="fts", limit=min(max(limit * 10, 50), 200))
    )
    weights = _term_weights(terms, [hit.text for hit in page.hits])
    ranked = sorted(
        ((-_page_score(hit.text, terms, weights), rank, hit) for rank, hit in enumerate(page.hits)),
    )
    citations: list[MemoryCitation] = []
    seen: set[str] = set()
    for _score, _rank, hit in ranked:
        if hit.entry_id in seen:
            continue
        seen.add(hit.entry_id)
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
        if len(citations) >= limit:
            break
    return tuple(citations)


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)

__all__ = [
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
