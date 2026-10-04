"""
tests/test_ai_service.py – Unit tests for the Phase 2 AI service module.

All tests work WITHOUT a real API key.  API calls are mocked.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import pathlib
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import ai_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adf():
    """Small analysis DataFrame with known values."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10", "2024-04-15"]),
        "product": ["A", "B", "A", "B"],
        "revenue": [100.0, 200.0, 300.0, 300.0],
    })


# ---------------------------------------------------------------------------
# Tool dispatch & validation
# ---------------------------------------------------------------------------

class TestToolDispatch(unittest.TestCase):

    def setUp(self):
        self.adf = _make_adf()

    def test_sales_summary_keys(self):
        r = ai_service.dispatch_tool("sales_summary", {}, self.adf, "date", "product", "revenue")
        for key in ("total_revenue", "avg_revenue_per_valid_row", "total_rows",
                     "valid_revenue_rows", "distinct_products"):
            self.assertIn(key, r)
        self.assertAlmostEqual(r["total_revenue"], 900.0)
        self.assertEqual(r["valid_revenue_rows"], 4)
        self.assertEqual(r["distinct_products"], 2)

    def test_top_products_limit(self):
        r = ai_service.dispatch_tool("top_products", {"limit": 1}, self.adf, "date", "product", "revenue")
        self.assertEqual(r["limit_used"], 1)
        self.assertEqual(len(r["products"]), 1)

    def test_top_products_clamp_high(self):
        r = ai_service.dispatch_tool("top_products", {"limit": 15}, self.adf, "date", "product", "revenue")
        self.assertEqual(r["limit_used"], 10)

    def test_top_products_clamp_low(self):
        r = ai_service.dispatch_tool("top_products", {"limit": 0}, self.adf, "date", "product", "revenue")
        self.assertEqual(r["limit_used"], 1)

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            ai_service.dispatch_tool("unknown_tool", {}, self.adf, "date", "product", "revenue")

    def test_data_quality_report(self):
        r = ai_service.dispatch_tool(
            "data_quality_report", {},
            self.adf, "date", "product", "revenue",
            original_df=self.adf, parse_info={"invalid_dates": 0}, remove_duplicates=False,
        )
        self.assertIn("original_quality_summary", r)
        self.assertIn("duplicate_policy", r)


# ---------------------------------------------------------------------------
# Monthly revenue analysis
# ---------------------------------------------------------------------------

class TestMonthlyRevenue(unittest.TestCase):

    def test_ties_and_changes(self):
        adf = _make_adf()  # Jan=300, Feb=300, Apr=300
        r = ai_service.dispatch_tool("monthly_revenue_analysis", {}, adf, "date", "product", "revenue")

        self.assertEqual(r["highest_revenue"], 300.0)
        self.assertEqual(r["lowest_revenue"], 300.0)
        # All three months tie
        self.assertEqual(len(r["highest_months"]), 3)
        self.assertEqual(len(r["lowest_months"]), 3)

        # Only Jan→Feb is consecutive
        changes = r["month_over_month_changes"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["from_month"], "2024-01")
        self.assertEqual(changes[0]["to_month"], "2024-02")
        self.assertAlmostEqual(changes[0]["change"], 0.0)
        self.assertAlmostEqual(changes[0]["pct_change"], 0.0)

    def test_gap_warning(self):
        adf = _make_adf()
        r = ai_service.dispatch_tool("monthly_revenue_analysis", {}, adf, "date", "product", "revenue")
        self.assertIn("gap_warning", r)
        self.assertIn("absent month", r["gap_warning"])

    def test_zero_denominator(self):
        adf = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-05", "2024-02-10"]),
            "product": ["A", "A"],
            "revenue": [0.0, 100.0],
        })
        r = ai_service.dispatch_tool("monthly_revenue_analysis", {}, adf, "date", "product", "revenue")
        changes = r["month_over_month_changes"]
        self.assertEqual(len(changes), 1)
        self.assertIsNone(changes[0]["pct_change"])
        self.assertIn("0", changes[0]["note"])

    def test_rows_excluded(self):
        adf = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-05", None]),
            "product": ["A", "B"],
            "revenue": [100.0, 200.0],
        })
        r = ai_service.dispatch_tool("monthly_revenue_analysis", {}, adf, "date", "product", "revenue")
        self.assertEqual(r["rows_excluded"], 1)


# ---------------------------------------------------------------------------
# Settings fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint(unittest.TestCase):

    def setUp(self):
        self.df = _make_adf()

    def test_same_inputs_same_hash(self):
        fp1 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", False, "English")
        fp2 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", False, "English")
        self.assertEqual(fp1, fp2)

    def test_filter_info_changes_hash(self):
        fp1 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", False, "English")
        filter_info = {"date_filtering_enabled": True, "start_date": "2024-01-01", "end_date": "2024-01-31"}
        fp2 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", False, "English", filter_info=filter_info)
        self.assertNotEqual(fp1, fp2)

    def test_different_language(self):
        fp1 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", False, "English")
        fp2 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", False, "Roman Urdu")
        self.assertNotEqual(fp1, fp2)

    def test_different_dedup(self):
        fp1 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", False, "English")
        fp2 = ai_service.compute_settings_fingerprint(self.df, "date", "product", "revenue", "auto", True, "English")
        self.assertNotEqual(fp1, fp2)


# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

class TestSystemInstruction(unittest.TestCase):

    def test_english_instruction(self):
        s = ai_service.build_system_instruction("English")
        self.assertIn("Never invent numbers", s)
        self.assertIn("Respond in English", s)

    def test_roman_urdu_instruction(self):
        s = ai_service.build_system_instruction("Roman Urdu")
        self.assertIn("Roman Urdu", s)
        self.assertIn("Never invent numbers", s)


# ---------------------------------------------------------------------------
# Missing key & API failure
# ---------------------------------------------------------------------------

class TestMissingKey(unittest.TestCase):

    @patch.dict(os.environ, {}, clear=False)
    def test_is_api_configured_false(self):
        with patch.object(ai_service, "get_api_key", return_value=None):
            self.assertFalse(ai_service.is_api_configured())

    def test_ask_question_no_key(self):
        with patch.object(ai_service, "get_api_key", return_value=None):
            r = ai_service.ask_question("test", _make_adf(), "date", "product", "revenue")
            self.assertIsNotNone(r["error"])
            self.assertIn("not configured", r["error"])

    def test_generate_report_no_key(self):
        with patch.object(ai_service, "get_api_key", return_value=None):
            r = ai_service.generate_report(_make_adf(), "date", "product", "revenue")
            self.assertIsNotNone(r["error"])


class TestAPIFailure(unittest.TestCase):

    @patch("ai_service.get_api_key", return_value="fake-key")
    @patch("ai_service.genai.Client" if hasattr(ai_service, "genai") else "google.genai.Client")
    @patch("ai_service.time.sleep", return_value=None)
    def test_api_exception_exhausted_retries(self, mock_sleep, mock_client_cls, _):
        """After 3 failed attempts due to 503, it should return the 'temporarily busy' error."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE")

        with patch.object(ai_service, "get_api_key", return_value="fake-key"):
            r = ai_service.ask_question("test", _make_adf(), "date", "product", "revenue")
        
        self.assertIsNotNone(r["error"])
        self.assertIn("temporarily busy", r["error"])
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

    @patch("ai_service.get_api_key", return_value="fake-key")
    @patch("ai_service.genai.Client" if hasattr(ai_service, "genai") else "google.genai.Client")
    @patch("ai_service.time.sleep", return_value=None)
    def test_api_exception_transient_recovery(self, mock_sleep, mock_client_cls, _):
        """It should retry on a 503 and succeed on the second attempt."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Turn 1: 503, then success
        mock_resp = MagicMock()
        mock_part = MagicMock(spec=[])
        mock_part.function_call = None
        mock_resp.candidates = [MagicMock()]
        mock_resp.candidates[0].content = MagicMock()
        mock_resp.candidates[0].content.parts = [mock_part]
        mock_resp.text = "Success after retry!"

        mock_client.models.generate_content.side_effect = [Exception("503 UNAVAILABLE"), mock_resp]

        with patch.object(ai_service, "get_api_key", return_value="fake-key"):
            r = ai_service.ask_question("test", _make_adf(), "date", "product", "revenue")
        
        self.assertIsNone(r["error"])
        self.assertEqual(r["answer"], "Success after retry!")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        
    @patch("ai_service.get_api_key", return_value="fake-key")
    @patch("ai_service.genai.Client" if hasattr(ai_service, "genai") else "google.genai.Client")
    def test_api_exception_fatal(self, mock_client_cls, _):
        """A fatal error like 400 Invalid Argument should NOT be retried."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("400 Invalid Argument")

        with patch.object(ai_service, "get_api_key", return_value="fake-key"):
            r = ai_service.ask_question("test", _make_adf(), "date", "product", "revenue")
        
        self.assertIsNotNone(r["error"])
        self.assertIn("400 Invalid Argument", r["error"])
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    @patch("ai_service.get_api_key", return_value="fake-key")
    @patch("ai_service.genai.Client" if hasattr(ai_service, "genai") else "google.genai.Client")
    def test_http_options_configuration(self, mock_client_cls, _):
        """Ensure the SDK client is configured with correct timeouts (ms) and attempts=1."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value.text = "OK"

        with patch.object(ai_service, "get_api_key", return_value="fake-key"):
            ai_service.ask_question("test", _make_adf(), "date", "product", "revenue")
        
        # Check ask_question client args
        kwargs = mock_client_cls.call_args[1]
        self.assertEqual(kwargs["http_options"].timeout, 15000)
        self.assertEqual(kwargs["http_options"].retry_options.attempts, 1)

        with patch.object(ai_service, "get_api_key", return_value="fake-key"):
            ai_service.generate_report(_make_adf(), "date", "product", "revenue")
            
        kwargs = mock_client_cls.call_args[1]
        self.assertEqual(kwargs["http_options"].timeout, 30000)
        self.assertEqual(kwargs["http_options"].retry_options.attempts, 1)


# ---------------------------------------------------------------------------
# Mocked successful tool-call flow
# ---------------------------------------------------------------------------

class TestMockedToolFlow(unittest.TestCase):

    def test_grounded_tool_call(self):
        """Mock a full function-call round trip."""
        adf = _make_adf()

        # Build mock objects for two generate_content calls
        # Turn 1: model requests sales_summary
        mock_fc = MagicMock()
        mock_fc.name = "sales_summary"
        mock_fc.args = {}

        mock_part1 = MagicMock()
        mock_part1.function_call = mock_fc

        mock_resp1 = MagicMock()
        mock_resp1.candidates = [MagicMock()]
        mock_resp1.candidates[0].content = MagicMock()
        mock_resp1.candidates[0].content.parts = [mock_part1]

        # Turn 2: model returns text
        mock_part2 = MagicMock()
        mock_part2.function_call = None
        mock_resp2 = MagicMock()
        mock_resp2.candidates = [MagicMock()]
        mock_resp2.candidates[0].content = MagicMock()
        mock_resp2.candidates[0].content.parts = [mock_part2]
        mock_resp2.text = "Total revenue is 900."

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [mock_resp1, mock_resp2]

        with patch.object(ai_service, "get_api_key", return_value="fake-key"), \
             patch("google.genai.Client", return_value=mock_client):
            r = ai_service.ask_question("What is total revenue?", adf, "date", "product", "revenue")

        self.assertIsNone(r["error"], f"Expected no error but got: {r.get('error')}")
        self.assertEqual(r["answer"], "Total revenue is 900.")
        self.assertEqual(len(r["tool_calls"]), 1)
        self.assertEqual(r["tool_calls"][0]["name"], "sales_summary")
        self.assertAlmostEqual(r["tool_calls"][0]["result"]["total_revenue"], 900.0)
        
        # Regression check: ensure the second request passes role="user", not role="tool"
        calls = mock_client.models.generate_content.call_args_list
        self.assertEqual(len(calls), 2)
        second_call_kwargs = calls[1][1]
        second_call_contents = second_call_kwargs["contents"]
        
        # contents should be: [user_question, model_function_call, user_function_response]
        self.assertEqual(len(second_call_contents), 3)
        self.assertEqual(second_call_contents[2].role, "user")


class TestReportMetadata(unittest.TestCase):
    def setUp(self):
        self.original_df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"]), "p": ["A"], "r": [100]})
        self.adf = self.original_df.copy()
        
    @patch("ai_service.get_api_key", return_value="fake")
    @patch("ai_service._generate_with_retry")
    def test_filters_disabled(self, mock_gen, mock_key):
        mock_gen.return_value.text = "Fake report"
        # Simulate generate_report with no filters or dupes removed
        res = ai_service.generate_report(
            self.adf, "d", "p", "r", 
            original_df=self.original_df,
            parse_info={"duplicates_removed": 0},
            remove_duplicates=False,
            filter_info=None  # or passed with disabled flags
        )
        md = res["report_markdown"]
        self.assertIn("### Analysis scope", md)
        self.assertIn("- Uploaded row count: 1", md)
        self.assertIn("- Rows removed by duplicate cleaning: 0", md)
        self.assertIn("- Rows before filtering (after duplicate policy): 1", md)
        self.assertIn("- Rows excluded by filters: 0", md)
        self.assertIn("- Rows after filtering: 1", md)
        self.assertIn("- Date filter: Disabled", md)
        self.assertIn("- Product filter: Disabled", md)

    @patch("ai_service.get_api_key", return_value="fake")
    @patch("ai_service._generate_with_retry")
    def test_filters_enabled(self, mock_gen, mock_key):
        mock_gen.return_value.text = "Fake report"
        filter_info = {
            "date_filtering_enabled": True,
            "start_date": pd.Timestamp("2024-01-01"),
            "end_date": pd.Timestamp("2024-01-31"),
            "product_filtering_enabled": True,
            "selected_products": ["A", "B"],
            "rows_before": 10,
            "rows_after": 7,
            "rows_excluded": 3,
        }
        res = ai_service.generate_report(
            self.adf, "d", "p", "r", 
            original_df=self.original_df, # e.g. length 12
            parse_info={"duplicates_removed": 2},
            remove_duplicates=True,
            filter_info=filter_info
        )
        md = res["report_markdown"]
        self.assertIn("### Analysis scope", md)
        self.assertIn("- Uploaded row count: 1", md) # From len(original_df)
        self.assertIn("- Rows removed by duplicate cleaning: 2", md)
        self.assertIn("- Rows before filtering (after duplicate policy): 10", md)
        self.assertIn("- Rows excluded by filters: 3", md)
        self.assertIn("- Rows after filtering: 7", md)
        self.assertIn("2024-01-01 to 2024-01-31", md)
        self.assertIn("- Product filter: A, B", md)

    @patch("ai_service.get_api_key", return_value="fake")
    @patch("ai_service._generate_with_retry")
    def test_duplicate_cleaning_and_filtering(self, mock_gen, mock_key):
        mock_gen.return_value.text = "Fake report"
        filter_info = {
            "date_filtering_enabled": False,
            "product_filtering_enabled": True,
            "selected_products": ["A"],
            "rows_before": 20,
            "rows_after": 15,
            "rows_excluded": 5,
        }
        res = ai_service.generate_report(
            self.adf, "d", "p", "r", 
            original_df=pd.DataFrame({"d": ["2024-01-01"] * 25, "p": ["A"] * 25, "r": [100] * 25}), 
            parse_info={"duplicates_removed": 5},
            remove_duplicates=True,
            filter_info=filter_info
        )
        md = res["report_markdown"]
        self.assertIn("### Analysis scope", md)
        self.assertIn("- Uploaded row count: 25", md)
        self.assertIn("- Rows removed by duplicate cleaning: 5", md)
        self.assertIn("- Rows before filtering (after duplicate policy): 20", md)
        self.assertIn("- Rows excluded by filters: 5", md)
        self.assertIn("- Rows after filtering: 15", md)
        self.assertIn("- Date filter: Disabled", md)
        self.assertIn("- Product filter: A", md)

if __name__ == "__main__":
    unittest.main()
