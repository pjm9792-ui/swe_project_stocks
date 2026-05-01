# Stocks Screening + Analyst Pipeline

This project is a two-stage stock research pipeline:

1. `stocks.py` builds a U.S. equity universe, refreshes cached market data, runs a momentum + insider-buying screen, and packages per-ticker research files.
2. `agent_workflow.py` takes a selected screener rank range and sends those ticker packages to the OpenAI API to generate structured analyst writeups.

`run_pipeline.py` is a convenience wrapper that runs both stages in sequence.

## Strategy

The idea is not "find random cheap stocks." The strategy is closer to:

- Start with U.S. listed common stocks and filter out ETFs, SPACs, ADRs, warrants, preferreds, and other non-core listings.
- Pull recent price history and look for names with strong medium-term momentum.
- Score stocks based on whether the recent half of the trend is cleaner and steeper than the older half.
- Overlay insider purchase activity from OpenInsider to catch names where management is also buying.
- Export a final union of:
  - technically strong names
  - names with unusually strong insider buying
- Build a research package per ticker with Yahoo Finance snapshots, price history, selected SEC filings, and chart artifacts.
- Feed those packages to an LLM analyst prompt that tries to answer: "Is this a real but still incomplete repricing, or is the move already priced in?"
- Optionally persist screening runs, analysis sessions, and stock reports into MongoDB when `MONGODB_URI` is configured.

This is a research workflow, not a production trading system or a validated alpha engine.

## Repo Structure

- `stocks.py`: market data refresh, screening logic, insider overlay, charts, and per-ticker package creation
- `agent_workflow.py`: per-ticker OpenAI analyst runs
- `run_pipeline.py`: one-command wrapper for both stages
- `requirements.txt`: Python dependencies

Generated artifacts are intentionally ignored from git:

- `data/`
- `screening_output/`
- `agents_data_package/`
- `output/`

## Setup

Create and activate a virtualenv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with at least:

```bash
OPENAI_API_KEY=your_key_here
```

Optional environment variables:

```bash
OPENAI_MODEL=gpt-5.4-mini
OPENAI_WEB_TOOL_TYPE=web_search_preview
SEC_USER_AGENT=YourName your_email@example.com
MONGODB_URI=your_mongodb_atlas_uri
MONGODB_DB_NAME=stocks_app
```

## Usage

Run only the stock screening / package builder:

```bash
python stocks.py
```

Run only the analyst workflow on screener ranks 1 through 3:

```bash
python agent_workflow.py --start-rank 1 --end-rank 3
```

Run the full pipeline end-to-end:

```bash
python run_pipeline.py --start-rank 1 --end-rank 3
```

Run the bare-bones Flask app:

```bash
python app.py
```

Useful options:

```bash
python stocks.py --refresh-metadata --refresh-insider
python run_pipeline.py --skip-stocks --start-rank 1 --end-rank 3
python agent_workflow.py --start-rank 10 --end-rank 20 --max-workers 2
```

## Outputs

`stocks.py` generates:

- ranked CSVs / feather files in `screening_output/`
- a PDF of top screening charts
- per-ticker package folders in `agents_data_package/`

`agent_workflow.py` generates:

- per-ticker raw API responses
- per-ticker text writeups
- parsed JSON summaries
- a combined session report text file
- a selected-range chart PDF bundle
- run manifests in `output/agent_runs/`

When MongoDB is configured, the app can also persist:

- global screening snapshots
- analysis sessions
- per-stock report records

The Flask app provides:

- register / login
- dashboard with rank-range selection
- live log polling while analysis runs
- results page with combined report text and saved chart artifacts

## Notes

- Yahoo Finance access through `yfinance` is unofficial and can be flaky.
- SEC downloads require a reasonable `SEC_USER_AGENT`.
- The LLM stage is optimized for research speed and structure, not guaranteed investment quality.

## Disclaimer

This project is for research and educational use. It is not investment advice.
