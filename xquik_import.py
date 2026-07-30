from __future__ import annotations

from collections.abc import Iterable
from html import escape

import pandas as pd

_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")
_SPREADSHEET_CONTROL_PREFIXES = ("\t", "\r", "\n")
TEXT_ALIASES = (
    "review",
    "reviews",
    "feedback",
    "comment",
    "comments",
    "message",
    "body",
    "text",
    "tweet text",
    "tweet_text",
    "tweettext",
)


def escape_review_html(review_text: object, *, max_length: int) -> str:
    """Truncate and encode review text before inserting it into HTML."""
    text = str(review_text)
    if len(text) > max_length:
        text = f"{text[:max_length]}…"
    return escape(text)


def export_review_csv(raw_df: pd.DataFrame, *, text_column: str) -> str:
    """Export reviews without leaving spreadsheet formulas executable."""
    export_df = raw_df.copy()
    export_df[text_column] = export_df[text_column].map(_neutralize_spreadsheet_formula)
    return export_df.to_csv(index=False)


def _neutralize_spreadsheet_formula(value: object) -> str:
    text = str(value)
    without_spaces = text.lstrip(" ")
    if text.lstrip().startswith(
        _SPREADSHEET_FORMULA_PREFIXES
    ) or without_spaces.startswith(_SPREADSHEET_CONTROL_PREFIXES):
        return f"'{text}"
    return text


def normalize_column_name(column_name: object) -> str:
    return "".join(
        character
        for character in str(column_name).strip().lower()
        if character.isalnum()
    )


def find_review_text_column(columns: Iterable[object]) -> str | None:
    normalized_aliases = {normalize_column_name(alias) for alias in TEXT_ALIASES}
    for column in columns:
        if normalize_column_name(column) in normalized_aliases:
            return str(column)
    return None


def detect_review_text_column(raw_df: pd.DataFrame) -> str | None:
    explicit_column = find_review_text_column(raw_df.columns)
    if explicit_column is not None:
        return explicit_column
    if raw_df.empty:
        return None
    average_lengths = {
        column: raw_df[column].fillna("").astype(str).str.len().mean()
        for column in raw_df.columns
    }
    if not average_lengths:
        return None
    return max(average_lengths, key=average_lengths.get)


def read_review_csv(source: object) -> pd.DataFrame:
    """Read a review CSV and report expected upload errors clearly."""
    try:
        raw_df = pd.read_csv(source)
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise ValueError("CSV must be a non-empty, valid UTF-8 file.") from error
    column_names = [str(column).strip() for column in raw_df.columns]
    if len(column_names) != len(set(column_names)):
        raise ValueError("CSV column names must be unique after trimming.")
    raw_df.columns = column_names
    return raw_df


def prepare_reviews(raw_df: pd.DataFrame) -> list[str]:
    text_column = detect_review_text_column(raw_df)
    if text_column is None:
        raise ValueError(
            "CSV must include a review, comment, text, or Tweet Text column."
        )
    reviews = raw_df[text_column].dropna().astype(str).str.strip()
    reviews = reviews[reviews != ""]
    if reviews.empty:
        raise ValueError("CSV must include at least one non-empty review row.")
    return reviews.tolist()
