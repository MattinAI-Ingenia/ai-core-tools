from dataclasses import dataclass, field
from typing import BinaryIO

import pandas as pd


@dataclass(frozen=True)
class ParsedRow:
    url: str
    row_metadata: dict = field(default_factory=dict)


def read_csv_headers(file_obj: BinaryIO) -> list[str]:
    file_obj.seek(0)
    df = pd.read_csv(file_obj, nrows=0)
    return list(df.columns)


def parse_and_dedupe_csv(file_obj: BinaryIO, link_column: str) -> tuple[list[ParsedRow], int]:
    file_obj.seek(0)
    df = pd.read_csv(file_obj, dtype=str, keep_default_na=False)

    if link_column not in df.columns:
        raise ValueError(f"link column '{link_column}' not found in CSV headers: {list(df.columns)}")

    metadata_columns = [c for c in df.columns if c != link_column]

    seen: dict[tuple, ParsedRow] = {}
    no_link_count = 0

    for _, record in df.iterrows():
        url = record[link_column].strip()
        if not url:
            no_link_count += 1
            continue

        row_metadata = {col: record[col] for col in metadata_columns}
        # Dedupe by URL alone: the resource's filename is derived from the URL
        # (see import_job_confirm._pdf_filename), so two rows sharing a link
        # would import the same PDF twice under the same name — the second
        # silently overwriting the first's file on disk. A shared PDF is one
        # document even if the CSV lists it under different metadata; the
        # first row's metadata wins.
        seen.setdefault(url, ParsedRow(url=url, row_metadata=row_metadata))

    return list(seen.values()), no_link_count
