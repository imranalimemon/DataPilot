"""
analysis.py – Pure calculation logic for DataPilot Phase 1.

Every public function takes a DataFrame (or Series) and returns results
without mutating the original data.  This keeps the analysis testable
and independent of the Streamlit UI.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Column-suggestion helpers
# ---------------------------------------------------------------------------

_DATE_HINTS = {"date", "order_date", "orderdate", "sale_date", "saledate",
               "transaction_date", "transactiondate", "day", "timestamp"}
_PRODUCT_HINTS = {"product", "product_name", "productname", "item",
                  "item_name", "itemname", "sku", "description"}
_REVENUE_HINTS = {"revenue", "sales", "total", "amount", "price",
                  "total_revenue", "totalrevenue", "total_sales",
                  "totalsales", "net_sales", "netsales", "income"}


def suggest_column(columns: list[str], hints: set[str]) -> str | None:
    """Return the first column whose lower-cased name is in *hints*."""
    for col in columns:
        if col.strip().lower().replace(" ", "_") in hints:
            return col
    return None


def suggest_date_column(columns: list[str]) -> str | None:
    return suggest_column(columns, _DATE_HINTS)


def suggest_product_column(columns: list[str]) -> str | None:
    return suggest_column(columns, _PRODUCT_HINTS)


def suggest_revenue_column(columns: list[str]) -> str | None:
    return suggest_column(columns, _REVENUE_HINTS)


# ---------------------------------------------------------------------------
# Data-quality helpers
# ---------------------------------------------------------------------------

def data_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a dict describing row counts, duplicates, and missing values."""
    total_rows = len(df)
    duplicate_rows = int(df.duplicated(keep=False).sum())
    exact_duplicate_rows = int(df.duplicated(keep="first").sum())
    missing_per_col = df.isnull().sum().to_dict()
    return {
        "total_rows": total_rows,
        "duplicate_rows_total": duplicate_rows,
        "exact_duplicate_rows": exact_duplicate_rows,
        "missing_per_column": missing_per_col,
    }


# ---------------------------------------------------------------------------
# Preparation helpers (return *copies*; never mutate the input)
# ---------------------------------------------------------------------------

def prepare_analysis_copy(
    df: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    date_format: str = "auto",
    remove_duplicates: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return (analysis_df, parse_info).

    *analysis_df* is a **copy** of *df* where:
    - The date column has been parsed (invalid → NaT).
    - The revenue column has been coerced to numeric (invalid → NaN).
    - Exact-duplicate rows are optionally removed.

    *parse_info* carries counts of invalid conversions so the UI can
    report them.
    """
    adf = df.copy()

    # --- Parse dates ---
    if date_format == "auto":
        adf[date_col] = pd.to_datetime(adf[date_col], errors="coerce", format="mixed")
    elif date_format == "dayfirst":
        adf[date_col] = pd.to_datetime(adf[date_col], errors="coerce", dayfirst=True, format="mixed")
    elif date_format == "monthfirst":
        adf[date_col] = pd.to_datetime(adf[date_col], errors="coerce", dayfirst=False, format="mixed")
    else:
        adf[date_col] = pd.to_datetime(adf[date_col], errors="coerce", format="mixed")

    invalid_dates = int(adf[date_col].isna().sum() - df[date_col].isna().sum())

    # --- Parse revenue ---
    adf[revenue_col] = pd.to_numeric(adf[revenue_col], errors="coerce")
    invalid_revenue = int(adf[revenue_col].isna().sum() - df[revenue_col].isna().sum())

    # --- Negative revenue ---
    negative_revenue_count = int((adf[revenue_col] < 0).sum())

    # --- Duplicates ---
    duplicates_removed = 0
    if remove_duplicates:
        before = len(adf)
        adf = adf.drop_duplicates(keep="first")
        duplicates_removed = before - len(adf)

    parse_info = {
        "invalid_dates": invalid_dates,
        "invalid_revenue": invalid_revenue,
        "negative_revenue_count": negative_revenue_count,
        "duplicates_removed": duplicates_removed,
    }
    return adf, parse_info


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def apply_filters(
    adf: pd.DataFrame,
    date_col: str,
    product_col: str,
    filter_dates: bool = False,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    filter_products: bool = False,
    selected_products: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply date and product filters to the analysis DataFrame.
    Returns (filtered_adf, filter_info).
    """
    fdf = adf.copy()
    filter_info = {
        "date_filtering_enabled": filter_dates,
        "product_filtering_enabled": filter_products,
        "start_date": start_date,
        "end_date": end_date,
        "selected_products": selected_products or [],
        "rows_before": len(adf),
        "rows_after": len(adf),
        "rows_excluded": 0,
    }

    if filter_dates and start_date is not None and end_date is not None:
        # Include rows where date is between start and end (inclusive)
        mask = (fdf[date_col] >= start_date) & (fdf[date_col] <= end_date)
        fdf = fdf[mask]
        
    if filter_products and selected_products is not None:
        # Include rows where product is in selected_products
        # Use isin, handling NaNs gracefully
        mask = fdf[product_col].isin(selected_products)
        fdf = fdf[mask]

    filter_info["rows_after"] = len(fdf)
    filter_info["rows_excluded"] = filter_info["rows_before"] - filter_info["rows_after"]
    return fdf, filter_info

# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def calc_total_revenue(adf: pd.DataFrame, revenue_col: str) -> tuple[float, int]:
    """Return (total_revenue, rows_used)."""
    valid = adf[revenue_col].dropna()
    return float(valid.sum()), len(valid)


def calc_avg_revenue(adf: pd.DataFrame, revenue_col: str) -> tuple[float, int]:
    """Return (average_revenue_per_valid_row, rows_used)."""
    valid = adf[revenue_col].dropna()
    if len(valid) == 0:
        return 0.0, 0
    return float(valid.mean()), len(valid)


def calc_distinct_products(adf: pd.DataFrame, product_col: str) -> int:
    """Count distinct non-empty product values."""
    products = adf[product_col].dropna()
    products = products[products.astype(str).str.strip() != ""]
    return int(products.nunique())


# ---------------------------------------------------------------------------
# Aggregations for charts / tables
# ---------------------------------------------------------------------------

def monthly_revenue(
    adf: pd.DataFrame, date_col: str, revenue_col: str
) -> pd.DataFrame:
    """
    Return a DataFrame with columns ``month`` and ``revenue``,
    sorted chronologically.  Rows missing date or revenue are excluded.
    """
    tmp = adf[[date_col, revenue_col]].dropna()
    if tmp.empty:
        return pd.DataFrame(columns=["month", "revenue"])
    tmp = tmp.copy()
    tmp["month"] = tmp[date_col].dt.to_period("M")
    grouped = tmp.groupby("month", sort=True)[revenue_col].sum().reset_index()
    grouped.columns = ["month", "revenue"]
    grouped["month"] = grouped["month"].astype(str)
    return grouped


def top_products_by_revenue(
    adf: pd.DataFrame, product_col: str, revenue_col: str, top_n: int = 10
) -> pd.DataFrame:
    """
    Return a DataFrame with columns ``product`` and ``revenue``
    for the top *top_n* products.  Rows missing product or revenue are excluded.
    """
    tmp = adf[[product_col, revenue_col]].dropna()
    # also exclude empty-string products
    tmp = tmp[tmp[product_col].astype(str).str.strip() != ""]
    if tmp.empty:
        return pd.DataFrame(columns=["product", "revenue"])
    grouped = (
        tmp.groupby(product_col)[revenue_col]
        .sum()
        .nlargest(top_n)
        .reset_index()
    )
    grouped.columns = ["product", "revenue"]
    return grouped
