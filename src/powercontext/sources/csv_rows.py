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

"""Row-level CSV capture with a stable identity and exact record evidence."""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
CSV_ROW_IDENTITY_PREFIX = "csv"
NonEmptyText = Annotated[str, Field(min_length=1)]


class CSVDocumentInput(BaseModel):
    """One offline CSV document captured row by row."""

    document_id: NonEmptyText
    primary_key: NonEmptyText
    content: bytes

    @field_validator("document_id")
    @classmethod
    def reject_padded_document_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
        return value


class CSVRowInput(BaseModel):
    """One captured CSV row with its document identity and raw field values."""

    document_id: NonEmptyText
    primary_key: NonEmptyText
    primary_key_value: NonEmptyText
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("columns")
    @classmethod
    def reject_empty_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("columns must not be empty")  # noqa: TRY003
        return value

    @field_validator("values")
    @classmethod
    def reject_width_mismatch(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        columns = info.data.get("columns")
        if columns is not None and len(columns) != len(value):
            raise ValueError("values must match the header width")  # noqa: TRY003
        return value


class CSVRowSource(Source):
    """Captured CSV row stored with the identity of its record."""

    source_id: NonEmptyText
    document_id: NonEmptyText
    primary_key: NonEmptyText
    primary_key_value: NonEmptyText
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Return the stable Source identity of one CSV row."""

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    if not isinstance(primary_key_value, str) or not primary_key_value:
        raise ValueError("primary_key_value must be a non-empty string")  # noqa: TRY003
    identity = f"{CSV_ROW_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(document: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse one offline CSV document into validated row-level captures."""

    document_id = document.document_id
    primary_key = document.primary_key
    if not primary_key:
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003

    try:
        text = document.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        records = list(reader)
    except csv.Error as error:
        raise ValueError(f"CSV content is malformed: {error}") from error

    if not records:
        raise ValueError("CSV document must contain a header row")  # noqa: TRY003
    header = records[0]
    if not header:
        raise ValueError("CSV header row must not be empty")  # noqa: TRY003
    _validate_header(header)
    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} is not a column of the document")  # noqa: TRY003
    key_index = header.index(primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(header):
            raise ValueError("CSV record width does not match the header width")  # noqa: TRY003
        key_value = record[key_index]
        if not key_value or not key_value.strip():
            raise ValueError("primary key value must not be empty or blank")  # noqa: TRY003
        if key_value in seen_keys:
            raise ValueError(f"primary key value {key_value!r} is duplicated")  # noqa: TRY003
        seen_keys.add(key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=key_value,
                columns=header,
                values=record,
            )
        )
    return tuple(rows)


def _validate_header(header: list[str]) -> None:
    seen: set[str] = set()
    for column in header:
        if not column or not column.strip():
            raise ValueError("CSV header fields must not be empty or blank")  # noqa: TRY003
        if column in seen:
            raise ValueError(f"CSV header field {column!r} is duplicated")  # noqa: TRY003
        seen.add(column)


class CSVRowSourceAdapter:
    """Resolve and read the row-level CSV Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one captured CSV row as textual evidence of its full record."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
