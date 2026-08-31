import unittest

import pandas as pd

from finBERT_model.daily_sentiment import aggregate_daily_sentiment


class DailySentimentTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = {
            "industrias": [
                {
                    "id": "industry-1",
                    "subindustrias": [
                        {"id": "sub-1", "nombre": "First"},
                        {"id": "sub-2", "nombre": "Second"},
                    ],
                }
            ]
        }

    def test_means_and_forward_fill_are_per_subindustry(self):
        news = pd.DataFrame(
            {
                "published_at": [
                    "2022-01-01T10:00:00Z",
                    "2022-01-01T11:00:00Z",
                    "2022-01-03T10:00:00Z",
                ],
                "subindustria_id": ["sub-1", "sub-1", "sub-2"],
                "sentiment_value": [1.0, -1.0, 0.5],
            }
        )

        result = aggregate_daily_sentiment(
            news, self.taxonomy, start_date="2022-01-01", end_date="2022-01-03"
        )
        sub_one = result[result["subindustria_id"] == "sub-1"]
        sub_two = result[result["subindustria_id"] == "sub-2"]

        self.assertEqual(len(result), 6)
        self.assertAlmostEqual(sub_one.iloc[0]["sentiment_mean"], 0.0)
        self.assertAlmostEqual(sub_one.iloc[1]["sentiment"], 0.0)
        self.assertAlmostEqual(sub_two.iloc[2]["sentiment"], 0.5)

    def test_empty_taxonomy_rows_are_retained(self):
        result = aggregate_daily_sentiment(
            pd.DataFrame(),
            self.taxonomy,
            start_date="2022-01-01",
            end_date="2022-01-02",
        )

        self.assertEqual(set(result["subindustria_id"]), {"sub-1", "sub-2"})
        self.assertTrue(result["sentiment"].isna().all())


if __name__ == "__main__":
    unittest.main()