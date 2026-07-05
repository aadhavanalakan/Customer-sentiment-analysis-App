from __future__ import annotations

import unittest

import pandas as pd

from xquik_import import detect_review_text_column, prepare_reviews


class XquikImportTests(unittest.TestCase):
    def test_prepare_reviews_prefers_xquik_tweet_text_header(self) -> None:
        raw_df = pd.DataFrame(
            {
                "Likes": [10, 11, 12],
                "Tweet Text": ["Great service", "  ", "Delivery was slow"],
            }
        )

        self.assertEqual(prepare_reviews(raw_df), ["Great service", "Delivery was slow"])

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


if __name__ == "__main__":
    unittest.main()
