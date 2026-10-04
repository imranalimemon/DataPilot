# DataPilot

**Turn sales data into clear business insights.**

DataPilot is a lightweight Streamlit dashboard that helps small businesses
upload a sales CSV, explore data quality, view key revenue metrics and
charts, ask AI-powered questions, and generate business reports — all
without writing code.

---

## Phase 1 features

| Feature | Description |
|---|---|
| CSV upload | Drag-and-drop a sales CSV (up to 10 MB) |
| Demo data | One-click synthetic dataset to explore the dashboard instantly |
| Data preview | First 20 rows displayed in a searchable table |
| Quality summary | Row count, exact duplicate count, missing values per column |
| Column mapping | Select date, product, and revenue columns; auto-suggestions with manual override |
| Date parsing | Choose automatic, day-first, or month-first parsing with a live preview |
| Duplicate handling | Optional checkbox to remove exact duplicates from the analysis copy (original data unchanged) |
| Analysis filters | Optional date range and product filtering to focus the analysis; syncs seamlessly with metrics and AI features |
| Revenue metrics | Total revenue, average revenue per valid row, distinct products, rows used |
| Monthly trend | Plotly bar chart of revenue per month, sorted chronologically |
| Top products | Plotly horizontal bar chart of top 10 products by revenue |
| Calculated tables | Monthly and product revenue totals displayed alongside charts |
| Missing values | Full column-by-column missing-value breakdown |

## Phase 2 features (AI-powered)

| Feature | Description |
|---|---|
| Ask your data | Type a question in English or Roman Urdu and get an AI-explained answer grounded in actual calculations |
| Gemini function calling | Questions are answered through real Gemini tool use — Python computes the numbers, Gemini explains them |
| Supported tools | `sales_summary`, `top_products`, `monthly_revenue_analysis`, `data_quality_report` |
| Evidence tables | Every AI answer is accompanied by the exact supporting calculation so users can verify numbers |
| Analysis steps | Collapsed section showing which tools were called and their parameters |
| Business report | One-click Gemini-written report with sections: Sales Overview, Product Performance, Monthly Trend, Data Quality, Suggested Next Actions |
| Report download | Download the generated report as a Markdown file |
| Grounding | All numerical results come from Python calculations; Gemini explains but never invents numbers |
| No-key fallback | Dashboard works fully without an API key; AI features show a setup message when key is absent |

### Synthetic demo data

The file `data/demo_sales.csv` is a small **synthetic** dataset created for
demonstration.  It includes several months of sales across multiple products
with intentional quirks (missing values, an invalid date, an exact duplicate
row) so that the quality-check features are visible.

### Expected CSV input

Your CSV should have at least three columns that can be mapped to:

- **Date** — transaction date in a recognisable format (e.g. `2024-01-15`)
- **Product** — product or item name
- **Revenue** — numeric sales value for the row

Additional columns are allowed and will be ignored by the analysis.

---

## Local setup

```bash
# 1. Clone / navigate to the project folder
cd DataPilot

# 2. Create a virtual environment and activate it
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the tests
python -m unittest discover tests -v

# 5. Launch the dashboard
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

### Setting up the Gemini API key (Phase 2)

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
3. Replace `YOUR_GEMINI_API_KEY_HERE` with your actual key.

Alternatively, set the environment variable:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-key-here"

# macOS / Linux
export GEMINI_API_KEY="your-key-here"
```

You can also set `GEMINI_MODEL` to use a different model (default: `gemini-3.8-flash`).

> **Note:** The dashboard works fully without an API key. AI features
> (Ask your data, Generate report) will show a setup message until a
> valid key is configured.

---

## Tech stack

- Python
- Streamlit
- Pandas
- Plotly
- Google Generative AI SDK (`google-genai`) — for Phase 2 AI features

No external API keys are required for Phase 1 features.

---

## Limitations

- The dashboard currently supports CSV files up to 10 MB in size.
- AI-generated explanations and insights are provided by Google Gemini and should be manually reviewed. While all numbers are grounded in deterministic Python calculations, the AI's narrative interpretations are generated contextually.
- In periods of high API demand, Gemini requests may temporarily time out. The dashboard will safely handle this and prompt you to try again shortly without crashing.
