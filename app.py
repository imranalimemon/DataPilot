"""
app.py – DataPilot: Sales Dashboard with AI-Powered Insights

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import pathlib
import streamlit as st
import pandas as pd
import plotly.express as px

import analysis
import ai_service

DEMO_PATH = pathlib.Path(__file__).parent / "data" / "demo_sales.csv"
MAX_UPLOAD_MB = 10


# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="DataPilot", layout="wide")
st.title("📊 DataPilot")
st.caption("Turn sales data into clear business insights.")


# ── Helpers ──────────────────────────────────────────────────────────────
def _fmt(value: float) -> str:
    """Format a number with two decimal places and thousands separators."""
    return f"{value:,.2f}"


@st.cache_data(show_spinner="Reading CSV …")
def load_csv(content: bytes) -> pd.DataFrame:
    """Try to read CSV bytes into a DataFrame with helpful error messages."""
    if len(content) == 0:
        raise ValueError("The uploaded file is empty.")
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                __import__("io").BytesIO(content),
                encoding=enc,
            )
            if df.columns.duplicated().any():
                dupes = list(df.columns[df.columns.duplicated(keep=False)].unique())
                raise ValueError(
                    f"Duplicate column names found: {dupes}. "
                    "Please rename them so every column is unique."
                )
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as exc:
            raise ValueError(f"Could not parse the CSV: {exc}") from exc
    raise ValueError(
        "Unable to read the file with UTF-8 or Latin-1 encoding. "
        "Please save it as UTF-8 CSV and try again."
    )


# ── Data source ──────────────────────────────────────────────────────────
st.subheader("Load your data")

uploaded = st.file_uploader(
    "Upload a sales CSV (max 10 MB)",
    type=["csv"],
)

use_demo = st.button("📂 Load demo data")

df: pd.DataFrame | None = None
using_demo = False

if uploaded is not None:
    raw = uploaded.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File exceeds {MAX_UPLOAD_MB} MB. Please upload a smaller file.")
        st.stop()
    try:
        df = load_csv(raw)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

elif use_demo or st.session_state.get("demo_loaded"):
    st.session_state["demo_loaded"] = True
    df = pd.read_csv(DEMO_PATH)
    using_demo = True

if df is None:
    st.info("Upload a CSV file or click **Load demo data** to get started.")
    # Show setup guidance for AI
    if not ai_service.is_api_configured():
        st.info("💡 **Tip:** To enable AI features after loading data, add your Gemini API key to `.streamlit/secrets.toml`.")
    st.stop()

if using_demo:
    st.info(
        "🔶 You are viewing **synthetic demo data** created for demonstration "
        "purposes. Load your own CSV to analyse real sales."
    )


# ── Data preview ─────────────────────────────────────────────────────────
with st.expander("Preview original data"):
    st.dataframe(df.head(20), use_container_width=True)


# ── Data quality ─────────────────────────────────────────────────────────
with st.expander("Original data quality"):
    quality = analysis.data_quality_summary(df)
    col_q1, col_q2, col_q3 = st.columns(3)
    col_q1.metric("Total rows", quality["total_rows"])
    col_q2.metric("Exact duplicate rows", quality["exact_duplicate_rows"])
    col_q3.metric(
        "Columns with missing values",
        sum(1 for v in quality["missing_per_column"].values() if v > 0),
    )

    missing = {k: v for k, v in quality["missing_per_column"].items() if v > 0}
    if missing:
        st.markdown("**Missing values per column:**")
        st.table(pd.DataFrame.from_dict(missing, orient="index", columns=["missing"]))
    else:
        st.success("No missing values found.")


# ── Advanced settings ────────────────────────────────────────────────────
with st.expander("Advanced settings (Mapping, parsing, duplicates)"):
    all_cols = list(df.columns)

    def _default_index(suggestion: str | None) -> int:
        if suggestion and suggestion in all_cols:
            return all_cols.index(suggestion)
        return 0

    col_m1, col_m2, col_m3 = st.columns(3)
    date_col = col_m1.selectbox(
        "Date column",
        all_cols,
        index=_default_index(analysis.suggest_date_column(all_cols)),
    )
    product_col = col_m2.selectbox(
        "Product column",
        all_cols,
        index=_default_index(analysis.suggest_product_column(all_cols)),
    )
    revenue_col = col_m3.selectbox(
        "Revenue column",
        all_cols,
        index=_default_index(analysis.suggest_revenue_column(all_cols)),
    )

    chosen = [date_col, product_col, revenue_col]
    if len(set(chosen)) < 3:
        st.error("Date, Product, and Revenue must be three **distinct** columns.")
        st.stop()

    st.markdown("**Date parsing**")
    date_fmt = st.radio(
        "How should dates be parsed?",
        ["auto", "dayfirst", "monthfirst"],
        horizontal=True,
        help=(
            "**auto** – let pandas guess. "
            "**dayfirst** – treat ambiguous dates as DD/MM/YYYY. "
            "**monthfirst** – treat ambiguous dates as MM/DD/YYYY."
        ),
    )

    preview_dates = pd.to_datetime(
        df[date_col], errors="coerce",
        dayfirst=(date_fmt == "dayfirst"),
        format="mixed",
    )
    st.caption("Parsed-date preview (first 5 rows):")
    st.dataframe(
        pd.DataFrame({"original": df[date_col].head(), "parsed": preview_dates.head()}),
        use_container_width=True,
    )

    remove_dupes = st.checkbox(
        "Remove exact duplicate rows from the analysis copy",
        value=False,
        help="The original data is never modified.",
    )


# ── Prepare analysis copy ────────────────────────────────────────────────
adf, parse_info = analysis.prepare_analysis_copy(
    df,
    date_col=date_col,
    revenue_col=revenue_col,
    date_format=date_fmt,
    remove_duplicates=remove_dupes,
)

notes: list[str] = []
if parse_info["invalid_dates"] > 0:
    notes.append(f"⚠️ {parse_info['invalid_dates']} date(s) could not be parsed and will be excluded from date-based charts.")
if parse_info["invalid_revenue"] > 0:
    notes.append(f"⚠️ {parse_info['invalid_revenue']} revenue value(s) could not be converted to numbers and are treated as missing.")
if parse_info["negative_revenue_count"] > 0:
    notes.append(f"ℹ️ {parse_info['negative_revenue_count']} row(s) have negative revenue (possibly refunds). These are included in calculations.")
if parse_info["duplicates_removed"] > 0:
    notes.append(f"🗑️ {parse_info['duplicates_removed']} exact duplicate row(s) removed from the analysis copy.")

if notes:
    for n in notes:
        if n.startswith("⚠"):
            st.warning(n)
        else:
            st.info(n)


# ── Analysis Filters ─────────────────────────────────────────────────────
st.subheader("Analysis filters")

if st.button("Reset filters"):
    for k in ["enable_date_filter", "enable_prod_filter", "selected_prods_filter"]:
        st.session_state.pop(k, None)
    st.rerun()

col_filt1, col_filt2 = st.columns(2)

with col_filt1:
    enable_date = st.checkbox("Filter by Date Range", key="enable_date_filter")
    valid_dates = adf[date_col].dropna()
    start_date, end_date = None, None
    if enable_date:
        if valid_dates.empty:
            st.warning("No valid dates found in the dataset.")
        else:
            min_d = valid_dates.min().date()
            max_d = valid_dates.max().date()
            # We don't use session_state for date_input default because we want it to react to date_col changes gracefully
            d_range = st.date_input("Date range (inclusive)", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if isinstance(d_range, tuple) and len(d_range) == 2:
                start_date = pd.to_datetime(d_range[0])
                end_date = pd.to_datetime(d_range[1])
            elif isinstance(d_range, tuple) and len(d_range) == 1:
                start_date = pd.to_datetime(d_range[0])
                end_date = start_date
            else:
                start_date = pd.to_datetime(d_range)
                end_date = start_date

with col_filt2:
    enable_prod = st.checkbox("Filter by Product", key="enable_prod_filter")
    selected_products = []
    if enable_prod:
        prod_choices = sorted(adf[product_col].dropna().astype(str).str.strip().unique())
        prod_choices = [p for p in prod_choices if p]
        selected_products = st.multiselect("Select products", prod_choices, key="selected_prods_filter")

adf, filter_info = analysis.apply_filters(
    adf, date_col, product_col,
    filter_dates=enable_date, start_date=start_date, end_date=end_date,
    filter_products=enable_prod, selected_products=selected_products
)

if enable_date or enable_prod:
    st.info(f"🔍 **Filters active:** {filter_info['rows_after']} rows matched "
               f"({filter_info['rows_excluded']} rows excluded).")
    if filter_info["rows_after"] == 0:
        st.error("No rows match the active filters.")


tab_dashboard, tab_ask, tab_report = st.tabs(["Dashboard", "Ask DataPilot", "Business Report"])

with tab_dashboard:
    total_rev, rev_rows = analysis.calc_total_revenue(adf, revenue_col)
    avg_rev, _ = analysis.calc_avg_revenue(adf, revenue_col)
    n_products = analysis.calc_distinct_products(adf, product_col)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total revenue", _fmt(total_rev))
    mc2.metric("Avg revenue per valid row", _fmt(avg_rev))
    mc3.metric("Distinct products", n_products)
    mc4.metric("Rows used (valid revenue)", rev_rows)

    st.caption(
        f"Revenue metrics use {rev_rows} of {len(adf)} rows in the analysis copy "
        "(rows with missing or unparseable revenue are excluded)."
    )

    monthly = analysis.monthly_revenue(adf, date_col, revenue_col)
    top_prods = analysis.top_products_by_revenue(adf, product_col, revenue_col)

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        if monthly.empty:
            st.warning("Not enough valid date + revenue data for a monthly chart.")
        else:
            excluded_trend = len(adf) - len(adf[[date_col, revenue_col]].dropna())
            if excluded_trend:
                st.caption(f"{excluded_trend} row(s) excluded (missing date or revenue).")
            fig_m = px.bar(
                monthly, x="month", y="revenue",
                labels={"month": "Month", "revenue": "Revenue"},
                title="Monthly revenue trend",
            )
            fig_m.update_layout(xaxis_type="category")
            st.plotly_chart(fig_m, use_container_width=True)
            with st.expander("Monthly totals"):
                st.dataframe(monthly, use_container_width=True, hide_index=True)

    with c_chart2:
        if top_prods.empty:
            st.warning("Not enough valid product + revenue data for a product chart.")
        else:
            tmp_prod = adf[[product_col, revenue_col]].dropna()
            excluded_prod = len(adf) - len(tmp_prod[tmp_prod[product_col].astype(str).str.strip() != ""])
            if excluded_prod:
                st.caption(f"{excluded_prod} row(s) excluded (missing or empty product, or missing revenue).")
            fig_p = px.bar(
                top_prods, x="revenue", y="product", orientation="h",
                labels={"product": "Product", "revenue": "Revenue"},
                title="Top 10 products by revenue",
            )
            fig_p.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_p, use_container_width=True)
            with st.expander("Product totals"):
                st.dataframe(top_prods, use_container_width=True, hide_index=True)

    with st.expander("Parsed analysis copy missing values"):
        st.caption("Note: Invalid dates and unparseable revenues are converted to missing values in this analysis copy.")
        miss_df = adf.isnull().sum().reset_index()
        miss_df.columns = ["column", "missing"]
        st.dataframe(miss_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 · AI-Powered Features
# ══════════════════════════════════════════════════════════════════════════

# ── Settings fingerprint (invalidate AI results when settings change) ────
_current_fp = ai_service.compute_settings_fingerprint(
    adf, date_col, product_col, revenue_col, date_fmt, remove_dupes,
    st.session_state.get("ai_language", "English"), filter_info=filter_info
)
if st.session_state.get("_settings_fp") != _current_fp:
    st.session_state["_settings_fp"] = _current_fp
    # Clear stale AI results
    for k in ("qa_result", "report_result"):
        st.session_state.pop(k, None)

_api_ok = ai_service.is_api_configured()

with tab_ask:
    if not _api_ok:
        st.info(
            "🔑 **Gemini API key not configured.** "
            "To enable AI features, copy `.streamlit/secrets.toml.example` "
            "to `.streamlit/secrets.toml` and add your API key."
        )

    ai_lang = st.radio(
        "Response language",
        ["English", "Roman Urdu"],
        horizontal=True,
        key="ai_language",
    )

    # Invalidate when language changes
    if st.session_state.get("_last_lang") != ai_lang:
        st.session_state["_last_lang"] = ai_lang
        st.session_state.pop("qa_result", None)

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    def _set_q(q: str):
        st.session_state["question_input"] = q

    st.caption("Example questions (click to populate):")
    ex_cols = st.columns(2)
    ex_cols[0].button("What is my total revenue?", on_click=_set_q, args=("What is my total revenue?",), use_container_width=True)
    ex_cols[0].button("Which products earn the most revenue?", on_click=_set_q, args=("Which products earn the most revenue?",), use_container_width=True)
    ex_cols[1].button("Which month had the highest revenue?", on_click=_set_q, args=("Which month had the highest revenue?",), use_container_width=True)
    ex_cols[1].button("What data-quality issues should I fix?", on_click=_set_q, args=("What data-quality issues should I fix?",), use_container_width=True)

    question = st.text_input(
        "Ask a question about your sales data",
        placeholder="e.g. What is my total revenue?",
        disabled=not _api_ok,
        key="question_input",
    )

    ask_clicked = st.button("🔍 Ask", disabled=not _api_ok or not question)

    if ask_clicked and question:
        st.caption("ℹ️ Your question and relevant sales summaries are sent to Google Gemini.")
        with st.spinner("Thinking …"):
            result = ai_service.ask_question(
                question=question,
                adf=adf,
                date_col=date_col,
                product_col=product_col,
                revenue_col=revenue_col,
                language=ai_lang,
                original_df=df,
                parse_info=parse_info,
                remove_duplicates=remove_dupes,
            )
            st.session_state["qa_result"] = result

    qa = st.session_state.get("qa_result")
    if qa:
        if qa.get("error"):
            st.error(f"⚠️ {qa['error']}")
        else:
            st.markdown(qa.get("answer", ""))

            # Show supporting evidence tables
            for tc in qa.get("tool_calls", []):
                res = tc.get("result", {})
                if tc["name"] == "sales_summary":
                    with st.expander("Supporting data — Sales Summary", expanded=True):
                        st.dataframe(
                            pd.DataFrame([{
                                "Total Revenue": res.get("total_revenue"),
                                "Avg Revenue": res.get("avg_revenue_per_valid_row"),
                                "Valid Rows": res.get("valid_revenue_rows"),
                                "Distinct Products": res.get("distinct_products"),
                            }]),
                            use_container_width=True, hide_index=True,
                        )

                elif tc["name"] == "top_products" and res.get("products"):
                    with st.expander("Supporting data — Top Products", expanded=True):
                        st.dataframe(
                            pd.DataFrame(res["products"]),
                            use_container_width=True, hide_index=True,
                        )
                        fig = px.bar(
                            pd.DataFrame(res["products"]),
                            x="revenue", y="product", orientation="h",
                            title="Top products by revenue",
                        )
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)

                elif tc["name"] == "monthly_revenue_analysis" and res.get("monthly_totals"):
                    with st.expander("Supporting data — Monthly Revenue", expanded=True):
                        mdf = pd.DataFrame(res["monthly_totals"])
                        st.dataframe(mdf, use_container_width=True, hide_index=True)
                        fig = px.bar(mdf, x="month", y="revenue", title="Monthly revenue trend")
                        fig.update_layout(xaxis_type="category")
                        st.plotly_chart(fig, use_container_width=True)

                elif tc["name"] == "data_quality_report":
                    with st.expander("Supporting data — Data Quality", expanded=True):
                        st.json(res)

            # Collapsed analysis steps
            with st.expander("Analysis steps"):
                for i, tc in enumerate(qa.get("tool_calls", []), 1):
                    st.markdown(f"**Step {i}:** Called `{tc['name']}`")
                    if tc.get("args"):
                        st.caption(f"Arguments: {json.dumps(tc['args'])}")


with tab_report:
    if not _api_ok:
        st.info(
            "🔑 **Gemini API key not configured.** "
            "To enable AI features, copy `.streamlit/secrets.toml.example` "
            "to `.streamlit/secrets.toml` and add your API key."
        )

    gen_clicked = st.button("📝 Generate report", disabled=not _api_ok)

    if gen_clicked:
        st.caption("ℹ️ Your question and relevant sales summaries are sent to Google Gemini.")
        with st.spinner("Generating report …"):
            report = ai_service.generate_report(
                adf=adf,
                date_col=date_col,
                product_col=product_col,
                revenue_col=revenue_col,
                language=ai_lang,
                original_df=df,
                parse_info=parse_info,
                remove_duplicates=remove_dupes,
                using_demo=using_demo,
                date_format=date_fmt,
                filter_info=filter_info,
            )
            st.session_state["report_result"] = report

    rpt = st.session_state.get("report_result")
    if rpt:
        if rpt.get("error"):
            st.error(f"⚠️ {rpt['error']}")
        else:
            st.download_button(
                "⬇️ Download report as Markdown",
                data=rpt.get("report_markdown", ""),
                file_name="datapilot_report.md",
                mime="text/markdown",
            )
            st.markdown(rpt.get("report_markdown", ""))
            
            with st.expander("Analysis steps"):
                for i, tc in enumerate(rpt.get("tool_calls", []), 1):
                    st.markdown(f"**Step {i}:** `{tc['name']}`")
                    if tc.get("args"):
                        st.caption(f"Arguments: {json.dumps(tc['args'])}")

