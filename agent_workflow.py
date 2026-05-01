import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages

from mongo_store import get_mongo_store

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SCREENING_FEATHER = BASE_DIR / "screening_output" / "final_screening_union.feather"
CHART_MANIFEST_CSV = BASE_DIR / "screening_output" / "chart_manifest.csv"
AGENTS_DATA_PACKAGE_DIR = BASE_DIR / "agents_data_package"
OUTPUT_ROOT = BASE_DIR / "output" / "agent_runs"

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

ANALYST_SYSTEM_PROMPT = """You are a stock analyst evaluating one ticker.

Decide if this is a REAL but INCOMPLETE repricing opportunity. The stock price has most likely increased a reasonable degree already, because this is a momentum strategy. 
Your goal is to figure out if the underlying fundamental catalyst is so large the the recent move still is not enough to price it in yet.  
Start from attached package files. 
Feel free to use web research, but also remember you already have some preliminary info uplaoded. 
If something is uncertain, use web search.
Be blunt, evidence-based, and decision-oriented. If financial models/valuation is necessary, do it.

Return exactly:
1) Executive Verdict (rating + stage + confidence)
2) What Changed
3) Why It Matters Economically
4) Evidence It Is Real (hard vs soft)
5) What Is Already Priced In
6) Remaining Uncertainty
7) Bull/Base/Bear (returns + probabilities summing to 100%)
8) Repricing vs Mean Reversion Probabilities
9) Disconfirming Signals (3-5)
10) Final Recommendation

End with:
FINAL_JSON:
{
  "ticker": "...",
  "rating": "Reject | Watchlist | Research Deeper | Buy Candidate",
  "repricing_stage": "Early | Middle | Late | Unclear",
  "confidence": "Low | Medium | High",
  "prob_upward_repricing": 0,
  "prob_mean_reversion": 0,
  "bull_case_return_pct": 0,
  "bull_case_prob": 0,
  "base_case_return_pct": 0,
  "base_case_prob": 0,
  "bear_case_return_pct": 0,
  "bear_case_prob": 0,
  "expected_return_pct": 0,
  "top_disconfirming_signals": ["", "", ""]
}
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-ticker analyst agents.")
    parser.add_argument(
        "--start-rank",
        type=int,
        default=1,
        help="Starting screener rank to analyze.",
    )
    parser.add_argument(
        "--end-rank",
        type=int,
        default=5,
        help="Ending screener rank to analyze.",
    )
    parser.add_argument("--max-workers", type=int, default=2, help="Concurrent per-ticker agent calls.")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument(
        "--web-tool-type",
        type=str,
        default=os.getenv("OPENAI_WEB_TOOL_TYPE", "web_search_preview"),
        help="Responses web search tool type.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id override.")
    parser.add_argument("--user-id", type=str, default="local-user", help="Application user id for session storage.")
    parser.add_argument("--user-email", type=str, default="", help="Optional user email for session storage.")
    parser.add_argument(
        "--include-feather",
        action="store_true",
        help="Include .feather files in uploaded package files (off by default).",
    )
    parser.add_argument(
        "--max-sec-html-files",
        type=int,
        default=4,
        help="Maximum number of SEC filing HTML files to include per ticker.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=1.5,
        help="Skip files larger than this size in MB to reduce context overflows.",
    )
    parser.add_argument(
        "--skip-tickers",
        type=str,
        default="",
        help="Comma-separated tickers to exclude (e.g., CMTV,ALMS,THM,HYMC,ASA).",
    )
    return parser.parse_args()


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (or OPENAI_KEY) in environment.")
    return key


def build_run_dirs(run_id: Optional[str]) -> Dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_id or ts
    root = OUTPUT_ROOT / run_name
    per_stock = root / "per_stock"
    charts = root / "charts"
    uploaded = root / "uploaded_file_manifest"
    for d in [root, per_stock, charts, uploaded]:
        d.mkdir(parents=True, exist_ok=True)
    return {"root": root, "per_stock": per_stock, "charts": charts, "uploaded": uploaded}


def resolve_rank_bounds(total_count: int, start_rank: int, end_rank: int, label: str) -> tuple[int, int]:
    if total_count <= 0:
        raise ValueError(f"No screened stocks are available for {label}.")

    normalized_start = max(1, int(start_rank))
    if normalized_start != int(start_rank):
        print(f"{label}: adjusted start rank from {start_rank} to {normalized_start}.")

    if normalized_start > total_count:
        raise ValueError(
            f"{label}: start rank {normalized_start} exceeds available screened stocks ({total_count})."
        )

    normalized_end = max(normalized_start, int(end_rank))
    if normalized_end > total_count:
        print(f"{label}: adjusted end rank from {normalized_end} to {total_count}.")
        normalized_end = total_count

    return normalized_start, normalized_end


def load_ranked_tickers(
    path: Path,
    start_rank: int,
    end_rank: int,
    skip_tickers: Optional[set] = None,
) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing screening feather: {path}")
    df = pd.read_feather(path)
    if "Ticker" not in df.columns:
        raise ValueError("Expected `Ticker` column in final_screening_union.feather.")
    if "combined_score" in df.columns:
        df = df.sort_values("combined_score", ascending=False, kind="stable")
    elif "technical_score" in df.columns:
        df = df.sort_values("technical_score", ascending=False, kind="stable")
    df = df.dropna(subset=["Ticker"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df = df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    df["screen_rank"] = range(1, len(df) + 1)

    start_rank, end_rank = resolve_rank_bounds(len(df), start_rank, end_rank, label="Analysis range")
    df = df[(df["screen_rank"] >= start_rank) & (df["screen_rank"] <= end_rank)].copy()
    if skip_tickers:
        df = df[~df["Ticker"].isin(skip_tickers)].copy()
    if df.empty:
        raise ValueError("No tickers found in screening feather.")
    return df.to_dict(orient="records")


def list_ticker_package_files(
    ticker: str,
    include_feather: bool,
    max_sec_html_files: int,
    max_file_size_mb: float,
) -> List[Path]:
    root = AGENTS_DATA_PACKAGE_DIR / ticker
    if not root.exists():
        raise FileNotFoundError(f"Missing ticker package directory: {root}")

    candidates = [
        root / "screening_snapshot.json",
        root / "price_history.csv",
        root / "price_history.feather",
        root / "package_meta.json",
        root / "yahoo" / "fast_info.json",
        root / "yahoo" / "info_selected.json",
        root / "yahoo" / "financials.csv",
        root / "yahoo" / "balance_sheet.csv",
        root / "yahoo" / "cashflow.csv",
        root / "yahoo" / "quarterly_financials.csv",
        root / "yahoo" / "quarterly_balance_sheet.csv",
        root / "yahoo" / "quarterly_cashflow.csv",
        root / "sec" / "selected_filings.csv",
    ]
    files = [p for p in candidates if p.exists()]
    sec_html = sorted((root / "sec" / "filings_html").glob("*.htm"))
    sec_html.extend(sorted((root / "sec" / "filings_html").glob("*.html")))
    if max_sec_html_files >= 0:
        sec_html = sec_html[:max_sec_html_files]
    files.extend(sec_html)

    if not include_feather:
        files = [p for p in files if p.suffix.lower() != ".feather"]
    if max_file_size_mb > 0:
        max_bytes = int(max_file_size_mb * 1024 * 1024)
        files = [p for p in files if p.stat().st_size <= max_bytes]
    return files


def http_post_json(
    session: requests.Session,
    url: str,
    headers: dict,
    payload: dict,
    timeout: int = 300,
    retries: int = 3,
) -> dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"{resp.status_code} {resp.text}")
            return resp.json()
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if "429" in msg or "rate_limit_exceeded" in msg:
                m = re.search(r"Please try again in ([0-9.]+)s", msg)
                wait_sec = float(m.group(1)) + 1.5 if m else (4.0 * attempt)
                time.sleep(wait_sec)
                continue
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"POST failed after {retries} attempts: {last_err}")


def upload_file(session: requests.Session, api_key: str, file_path: Path, retries: int = 3) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with open(file_path, "rb") as f:
                resp = session.post(
                    f"{OPENAI_BASE_URL}/files",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={"purpose": "user_data"},
                    files={"file": (file_path.name, f)},
                    timeout=180,
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"{resp.status_code} {resp.text}")
            body = resp.json()
            file_id = body.get("id")
            if not file_id:
                raise RuntimeError(f"No file id returned for {file_path}")
            return file_id
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"File upload failed for {file_path}: {last_err}")


def extract_response_text(resp_json: dict) -> str:
    if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
        return resp_json["output_text"]

    chunks: List[str] = []
    for item in resp_json.get("output", []) or []:
        content = item.get("content", []) if isinstance(item, dict) else []
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") in {"output_text", "text"}:
                txt = c.get("text", "")
                if txt:
                    chunks.append(txt)
    return "\n\n".join(chunks).strip()


def extract_final_json_block(text: str) -> Optional[dict]:
    marker = "FINAL_JSON:"
    idx = text.find(marker)
    if idx == -1:
        return None
    tail = text[idx + len(marker) :]
    start = tail.find("{")
    if start == -1:
        return None
    snippet = tail[start:]

    depth = 0
    in_str = False
    esc = False
    end_pos = None
    for i, ch in enumerate(snippet):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break
    if not end_pos:
        return None
    raw = snippet[:end_pos]
    try:
        return json.loads(raw)
    except Exception:
        return None


def build_analyst_user_prompt(ticker: str, package_files: List[Path]) -> str:
    rels = []
    for p in package_files:
        try:
            rels.append(str(p.relative_to(AGENTS_DATA_PACKAGE_DIR / ticker)))
        except Exception:
            rels.append(p.name)
    manifest = "\n".join(f"- {r}" for r in rels)
    return (
        f"Analyze ticker: {ticker}\n\n"
        "Use the attached files as primary evidence. Web search is allowed for missing/stale data.\n"
        "Be explicit about what changed, what is priced in, and expected value.\n\n"
        "Attached file manifest:\n"
        f"{manifest}\n\n"
        "Return the exact required section structure and include FINAL_JSON at the end."
    )


def create_response(
    session: requests.Session,
    api_key: str,
    model: str,
    reasoning_effort: str,
    web_tool_type: str,
    system_prompt: str,
    user_text: str,
    file_ids: List[str],
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    content = [{"type": "input_text", "text": user_text}]
    content.extend([{"type": "input_file", "file_id": fid} for fid in file_ids])

    payload = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "tools": [{"type": web_tool_type}],
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
    }
    try:
        return http_post_json(session, f"{OPENAI_BASE_URL}/responses", headers=headers, payload=payload)
    except Exception as first_exc:
        if web_tool_type != "web_search":
            payload["tools"] = [{"type": "web_search"}]
            return http_post_json(session, f"{OPENAI_BASE_URL}/responses", headers=headers, payload=payload)
        raise RuntimeError(str(first_exc))


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def create_session_chart_bundle(run_dirs: Dict[str, Path], selected_rows: List[Dict[str, object]]) -> Dict[str, object]:
    chart_paths = []
    if CHART_MANIFEST_CSV.exists():
        chart_df = pd.read_csv(CHART_MANIFEST_CSV)
        if {"Ticker", "chart_image_path"}.issubset(chart_df.columns):
            chart_df["Ticker"] = chart_df["Ticker"].astype(str).str.upper()
            selected_tickers = [str(row["Ticker"]).upper() for row in selected_rows]
            chart_paths = (
                chart_df[chart_df["Ticker"].isin(selected_tickers)]
                .sort_values("screen_rank" if "screen_rank" in chart_df.columns else "Ticker")
                ["chart_image_path"]
                .dropna()
                .astype(str)
                .tolist()
            )

    pdf_path = run_dirs["charts"] / "selected_screening_charts.pdf"
    included = []
    if chart_paths:
        with PdfPages(pdf_path) as pdf:
            for chart_path in chart_paths:
                image_path = Path(chart_path)
                if not image_path.exists():
                    continue
                img = mpimg.imread(image_path)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                included.append(str(image_path))

    return {
        "chart_pdf_path": str(pdf_path) if included else "",
        "chart_image_paths": included,
    }


def run_one_ticker(
    ticker_row: Dict[str, object],
    run_dirs: Dict[str, Path],
    api_key: str,
    model: str,
    reasoning_effort: str,
    web_tool_type: str,
    include_feather: bool,
    max_sec_html_files: int,
    max_file_size_mb: float,
) -> dict:
    ticker = str(ticker_row["Ticker"]).upper()
    started = time.time()
    out = {
        "ticker": ticker,
        "screen_rank": int(ticker_row.get("screen_rank", -1)),
        "status": "ok",
        "error": "",
        "n_files_uploaded": 0,
        "elapsed_sec": None,
        "text_path": "",
        "json_path": "",
        "raw_path": "",
        "parsed_report": None,
    }
    ticker_dir = run_dirs["per_stock"] / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    try:
        package_files = list_ticker_package_files(
            ticker,
            include_feather=include_feather,
            max_sec_html_files=max_sec_html_files,
            max_file_size_mb=max_file_size_mb,
        )
        if not package_files:
            raise RuntimeError(f"No package files found for {ticker}")

        session = requests.Session()
        uploaded = []
        for path in package_files:
            fid = upload_file(session, api_key, path)
            uploaded.append({"path": str(path), "file_id": fid})

        out["n_files_uploaded"] = len(uploaded)
        (run_dirs["uploaded"] / f"{ticker}_uploaded_files.json").write_text(
            json.dumps(uploaded, indent=2), encoding="utf-8"
        )

        user_prompt = build_analyst_user_prompt(ticker, package_files)
        try:
            resp = create_response(
                session=session,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
                web_tool_type=web_tool_type,
                system_prompt=ANALYST_SYSTEM_PROMPT,
                user_text=user_prompt,
                file_ids=[x["file_id"] for x in uploaded],
            )
        except Exception as exc:
            if "context_length_exceeded" not in str(exc):
                raise
            # Retry with SEC HTML removed when context is too large.
            reduced_ids = [
                x["file_id"]
                for x in uploaded
                if not str(x["path"]).lower().endswith(".htm") and not str(x["path"]).lower().endswith(".html")
            ]
            if not reduced_ids:
                raise
            resp = create_response(
                session=session,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
                web_tool_type=web_tool_type,
                system_prompt=ANALYST_SYSTEM_PROMPT,
                user_text=user_prompt + "\n\nNote: SEC filing HTML attachments were removed due to context limits.",
                file_ids=reduced_ids,
            )

        raw_path = ticker_dir / f"{ticker}.response_raw.json"
        txt_path = ticker_dir / f"{ticker}.txt"
        json_path = ticker_dir / f"{ticker}.json"
        raw_path.write_text(json.dumps(resp, indent=2), encoding="utf-8")
        text = extract_response_text(resp)
        txt_path.write_text(text, encoding="utf-8")
        parsed = extract_final_json_block(text)
        if parsed is not None:
            json_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        out["text_path"] = str(txt_path)
        out["json_path"] = str(json_path) if parsed is not None else ""
        out["raw_path"] = str(raw_path)
        out["parsed_report"] = parsed
    except Exception as exc:
        out["status"] = "error"
        out["error"] = str(exc)
        (ticker_dir / f"{ticker}.error.txt").write_text(str(exc), encoding="utf-8")
    finally:
        out["elapsed_sec"] = round(time.time() - started, 2)
    return out


def concatenate_ticker_reports(run_dirs: Dict[str, Path], selected_rows: List[Dict[str, object]]) -> Dict[str, object]:
    blocks = []
    included = []
    for row in selected_rows:
        ticker = str(row["Ticker"]).upper()
        rank = int(row["screen_rank"])
        txt_path = run_dirs["per_stock"] / ticker / f"{ticker}.txt"
        if not txt_path.exists():
            continue
        blocks.append(f"===== Rank {rank} | {ticker} =====\n\n{txt_path.read_text(encoding='utf-8')}")
        included.append(ticker)
    combined_text = "\n\n".join(blocks)
    output_path = run_dirs["root"] / "combined_per_stock_reports.txt"
    output_path.write_text(combined_text, encoding="utf-8")
    return {"combined_text": combined_text, "combined_path": str(output_path), "included_tickers": included}


def main() -> None:
    args = parse_args()
    api_key = get_api_key()
    mongo = get_mongo_store()
    if mongo is not None and args.user_id:
        mongo.upsert_user(args.user_id, email=(args.user_email or None))

    run_dirs = build_run_dirs(args.run_id)

    skip_tickers = {t.strip().upper() for t in args.skip_tickers.split(",") if t.strip()}
    selected_rows = load_ranked_tickers(
        SCREENING_FEATHER,
        start_rank=max(1, int(args.start_rank)),
        end_rank=max(1, int(args.end_rank)),
        skip_tickers=skip_tickers,
    )
    tickers = [str(row["Ticker"]).upper() for row in selected_rows]
    chart_bundle = create_session_chart_bundle(run_dirs, selected_rows)
    session_key = run_dirs["root"].name
    mongo_session_id = None
    if mongo is not None:
        mongo_session_id = mongo.create_analysis_session(
            make_json_safe(
                {
                    "session_key": session_key,
                    "user_id": args.user_id,
                    "user_email": args.user_email or None,
                    "run_dir": str(run_dirs["root"]),
                    "start_rank": int(args.start_rank),
                "end_rank": int(args.end_rank),
                "selected_tickers": tickers,
                "selected_rows": selected_rows,
                    "chart_artifacts": chart_bundle,
                    "status": "running",
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                }
            )
        )
    print(f"Run folder: {run_dirs['root']}")
    print(
        f"Tickers ({len(tickers)}) | ranks {int(args.start_rank)}-{int(args.end_rank)}: "
        f"{', '.join(tickers)}"
    )

    worker_count = min(max(1, args.max_workers), len(tickers))
    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_one_ticker,
                ticker_row=row,
                run_dirs=run_dirs,
                api_key=api_key,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                web_tool_type=args.web_tool_type,
                include_feather=bool(args.include_feather),
                max_sec_html_files=int(args.max_sec_html_files),
                max_file_size_mb=float(args.max_file_size_mb),
            )
            for row in selected_rows
        ]
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            print(
                f"[rank {item['screen_rank']} | {item['ticker']}] status={item['status']} files={item['n_files_uploaded']} "
                f"elapsed={item['elapsed_sec']}s"
            )
            if mongo is not None and mongo_session_id is not None:
                mongo.upsert_stock_report(
                    mongo_session_id,
                    item["ticker"],
                    {
                        "screen_rank": item["screen_rank"],
                        "status": item["status"],
                        "error": item["error"],
                        "text_path": item["text_path"],
                        "json_path": item["json_path"],
                        "raw_path": item["raw_path"],
                        "report_json": item["parsed_report"],
                    },
                )

    manifest_path = run_dirs["root"] / "run_manifest.json"
    combined = concatenate_ticker_reports(run_dirs, selected_rows)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "start_rank": args.start_rank,
        "end_rank": args.end_rank,
        "tickers": tickers,
        "selected_rows": selected_rows,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "web_tool_type": args.web_tool_type,
        "chart_bundle": chart_bundle,
        "combined_reports_path": combined["combined_path"],
        "results": sorted(results, key=lambda x: x["ticker"]),
    }
    manifest_path.write_text(json.dumps(make_json_safe(manifest), indent=2), encoding="utf-8")

    ok_tickers = [r["ticker"] for r in results if r["status"] == "ok"]
    if not ok_tickers:
        if mongo is not None and mongo_session_id is not None:
            mongo.update_analysis_session(
                mongo_session_id,
            {
                "status": "error",
                "manifest_path": str(manifest_path),
                "combined_reports_path": combined["combined_path"],
                "results": make_json_safe(results),
            },
        )
        raise RuntimeError("All per-ticker agent runs failed.")
    if mongo is not None and mongo_session_id is not None:
        mongo.update_analysis_session(
            mongo_session_id,
            {
                "status": "completed",
                "manifest_path": str(manifest_path),
                "combined_reports_path": combined["combined_path"],
                "included_tickers": combined["included_tickers"],
                "successful_tickers": sorted(ok_tickers),
                "results": make_json_safe(results),
            },
        )
    print(f"Saved run manifest: {manifest_path}")
    print(f"Saved combined reports: {combined['combined_path']}")
    if chart_bundle["chart_pdf_path"]:
        print(f"Saved selected chart pdf: {chart_bundle['chart_pdf_path']}")
    print(f"Successful per-ticker reports: {len(ok_tickers)} ({', '.join(sorted(ok_tickers))})")


if __name__ == "__main__":
    main()
