"""
tests/test_analysis.py – Unit tests for the analysis module.

Run with:  python -m pytest tests/ -v
  or:      python -m unittest discover tests -v
"""

import unittest
import pandas as pd
import numpy as np
import sys
import pathlib

# Ensure the project root is on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import analysis


class TestColumnSuggestions(unittest.TestCase):
    def test_suggest_date(self):
        self.assertEqual(analysis.suggest_date_column(["id", "date", "amt"]), "date")

    def test_suggest_product(self):
        self.assertEqual(analysis.suggest_product_column(["date", "Product", "rev"]), "Product")

    def test_suggest_revenue(self):
        self.assertEqual(analysis.suggest_revenue_column(["date", "item", "revenue"]), "revenue")

    def test_no_match(self):
        self.assertIsNone(analysis.suggest_date_column(["foo", "bar"]))


class TestDataQuality(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "a": [1, 2, 2, None],
            "b": ["x", "y", "y", "z"],
        })

    def test_total_rows(self):
        q = analysis.data_quality_summary(self.df)
        self.assertEqual(q["total_rows"], 4)

    def test_exact_duplicates(self):
        q = analysis.data_quality_summary(self.df)
        # rows index 1 and 2 are identical → 1 exact duplicate (keep="first")
        self.assertEqual(q["exact_duplicate_rows"], 1)

    def test_missing_values(self):
        q = analysis.data_quality_summary(self.df)
        self.assertEqual(q["missing_per_column"]["a"], 1)
        self.assertEqual(q["missing_per_column"]["b"], 0)


class TestPrepareAnalysisCopy(unittest.TestCase):
    """Use a tiny fixture with known values."""

    def setUp(self):
        self.df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01", "bad-date", "2024-03-01", "2024-01-01"],
            "product": ["A", "B", "A", "B", "A"],
            "revenue": ["100", "200", "50", "xyz", "100"],
        })

    def test_input_unchanged(self):
        """The original DataFrame must not be mutated."""
        original = self.df.copy()
        analysis.prepare_analysis_copy(self.df, "date", "revenue")
        pd.testing.assert_frame_equal(self.df, original)

    def test_invalid_dates_counted(self):
        _, info = analysis.prepare_analysis_copy(self.df, "date", "revenue")
        self.assertEqual(info["invalid_dates"], 1)  # "bad-date"

    def test_invalid_revenue_counted(self):
        _, info = analysis.prepare_analysis_copy(self.df, "date", "revenue")
        self.assertEqual(info["invalid_revenue"], 1)  # "xyz"

    def test_duplicates_removed(self):
        # Rows 0 and 4 are identical when parsed
        adf, info = analysis.prepare_analysis_copy(
            self.df, "date", "revenue", remove_duplicates=True
        )
        self.assertEqual(info["duplicates_removed"], 1)
        self.assertEqual(len(adf), 4)

    def test_duplicates_kept(self):
        adf, info = analysis.prepare_analysis_copy(
            self.df, "date", "revenue", remove_duplicates=False
        )
        self.assertEqual(info["duplicates_removed"], 0)
        self.assertEqual(len(adf), 5)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.adf = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "product": ["A", "B", "A"],
            "revenue": [100.0, 200.0, 150.0],
        })

    def test_total_revenue(self):
        total, rows = analysis.calc_total_revenue(self.adf, "revenue")
        self.assertAlmostEqual(total, 450.0)
        self.assertEqual(rows, 3)

    def test_avg_revenue(self):
        avg, rows = analysis.calc_avg_revenue(self.adf, "revenue")
        self.assertAlmostEqual(avg, 150.0)
        self.assertEqual(rows, 3)

    def test_distinct_products(self):
        n = analysis.calc_distinct_products(self.adf, "product")
        self.assertEqual(n, 2)

    def test_total_revenue_with_nan(self):
        adf = self.adf.copy()
        adf.loc[1, "revenue"] = np.nan
        total, rows = analysis.calc_total_revenue(adf, "revenue")
        self.assertAlmostEqual(total, 250.0)
        self.assertEqual(rows, 2)


class TestAggregations(unittest.TestCase):
    def setUp(self):
        self.adf = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-05", "2024-01-20", "2024-02-10", "2024-02-15"
            ]),
            "product": ["A", "B", "A", "B"],
            "revenue": [100.0, 200.0, 150.0, 250.0],
        })

    def test_monthly_revenue(self):
        m = analysis.monthly_revenue(self.adf, "date", "revenue")
        self.assertEqual(list(m["month"]), ["2024-01", "2024-02"])
        self.assertAlmostEqual(m.loc[m["month"] == "2024-01", "revenue"].iloc[0], 300.0)
        self.assertAlmostEqual(m.loc[m["month"] == "2024-02", "revenue"].iloc[0], 400.0)

    def test_top_products(self):
        t = analysis.top_products_by_revenue(self.adf, "product", "revenue")
        self.assertEqual(list(t["product"]), ["B", "A"])
        self.assertAlmostEqual(t.loc[t["product"] == "B", "revenue"].iloc[0], 450.0)
        self.assertAlmostEqual(t.loc[t["product"] == "A", "revenue"].iloc[0], 250.0)
        
        
class TestFilters(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", None]),
            "product": ["A", "B", "A", None],
            "revenue": [10.0, 20.0, 30.0, 40.0]
        })

    def test_original_unchanged(self):
        orig = self.df.copy()
        analysis.apply_filters(self.df, "date", "product", filter_dates=True, start_date=pd.to_datetime("2024-01-01"), end_date=pd.to_datetime("2024-01-31"))
        pd.testing.assert_frame_equal(self.df, orig)

    def test_inclusive_date_boundaries(self):
        start = pd.to_datetime("2024-01-01")
        end = pd.to_datetime("2024-01-15")
        fdf, info = analysis.apply_filters(self.df, "date", "product", filter_dates=True, start_date=start, end_date=end)
        self.assertEqual(len(fdf), 2)
        self.assertEqual(info["rows_after"], 2)

    def test_missing_retained_when_off(self):
        # When date filter is off, None dates should be retained
        fdf, _ = analysis.apply_filters(self.df, "date", "product", filter_dates=False)
        self.assertEqual(len(fdf), 4)

        # When date filter is on, None dates drop
        fdf, _ = analysis.apply_filters(self.df, "date", "product", filter_dates=True, start_date=pd.to_datetime("2000-01-01"), end_date=pd.to_datetime("2030-01-01"))
        self.assertEqual(len(fdf), 3)

    def test_combined_filters(self):
        fdf, _ = analysis.apply_filters(
            self.df, "date", "product",
            filter_dates=True, start_date=pd.to_datetime("2024-01-01"), end_date=pd.to_datetime("2024-12-31"),
            filter_products=True, selected_products=["B"]
        )
        self.assertEqual(len(fdf), 1)
        self.assertEqual(fdf.iloc[0]["product"], "B")

    def test_empty_matches(self):
        fdf, _ = analysis.apply_filters(self.df, "date", "product", filter_products=True, selected_products=["XYZ"])
        self.assertEqual(len(fdf), 0)


if __name__ == "__main__":
    unittest.main()
