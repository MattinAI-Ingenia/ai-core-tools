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
        dedupe_key = (url, tuple(sorted(row_metadata.items())))
        seen.setdefault(dedupe_key, ParsedRow(url=url, row_metadata=row_metadata))

    return list(seen.values()), no_link_count
