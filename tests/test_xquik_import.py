from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from xquik_import import (
    detect_review_text_column,
    escape_review_html,
    export_review_csv,
    prepare_reviews,
    read_review_csv,
)


class XquikImportTests(unittest.TestCase):
    def test_escape_review_html_encodes_markup(self) -> None:
        escaped = escape_review_html(
            '<img src=x onerror="alert(1)">&',
            max_length=100,
        )

        self.assertEqual(
            escaped,
            "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;",
        )

    def test_escape_review_html_truncates_before_encoding(self) -> None:
        self.assertEqual(escape_review_html("<>&", max_length=2), "&lt;&gt;…")

    def test_export_review_csv_neutralizes_spreadsheet_formulas(self) -> None:
        exported = export_review_csv(
            pd.DataFrame(
                {
                    "review": [
                        '=HYPERLINK("https://example.com")',
                        " +SUM(1,2)",
                        "-2+3",
                        "@SUM(1,2)",
                        "\tcommand",
                        " \rcommand",
                        "Ordinary review",
                    ]
                }
            ),
            text_column="review",
        )
        round_trip = pd.read_csv(BytesIO(exported.encode()))

        self.assertEqual(
            round_trip["review"].tolist(),
            [
                '\'=HYPERLINK("https://example.com")',
                "' +SUM(1,2)",
                "'-2+3",
                "'@SUM(1,2)",
                "'\tcommand",
                "' \rcommand",
                "Ordinary review",
            ],
        )

    def test_prepare_reviews_prefers_xquik_tweet_text_header(self) -> None:
        raw_df = pd.DataFrame(
            {
                "Likes": [10, 11, 12],
                "Tweet Text": ["Great service", "  ", "Delivery was slow"],
            }
        )

        self.assertEqual(
            prepare_reviews(raw_df), ["Great service", "Delivery was slow"]
        )

    def test_detect_review_text_column_uses_alias_before_longest_column(self) -> None:
        raw_df = pd.DataFrame(
            {
                "metadata": ["this metadata sentence is longer than the review"],
                "comment": ["good"],
            }
        )

        self.assertEqual(detect_review_text_column(raw_df), "comment")

    def test_prepare_reviews_rejects_empty_text_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            prepare_reviews(pd.DataFrame({"Tweet Text": [" ", None]}))

    def test_read_review_csv_rejects_empty_upload(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty, valid UTF-8"):
            read_review_csv(BytesIO(b""))

    def test_read_review_csv_rejects_malformed_upload(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty, valid UTF-8"):
            read_review_csv(BytesIO(b'comment\n"unterminated'))

    def test_read_review_csv_rejects_non_utf8_upload(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty, valid UTF-8"):
            read_review_csv(BytesIO(b"comment\n\xff\n"))

    def test_read_review_csv_rejects_duplicate_trimmed_headers(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique after trimming"):
            read_review_csv(BytesIO(b"Tweet Text, Tweet Text \nGood,Slow\n"))

    def test_read_review_csv_trims_headers(self) -> None:
        raw_df = read_review_csv(BytesIO(b" Tweet Text ,Likes\nGreat service,3\n"))

        self.assertEqual(raw_df.columns.tolist(), ["Tweet Text", "Likes"])


if __name__ == "__main__":
    unittest.main()
