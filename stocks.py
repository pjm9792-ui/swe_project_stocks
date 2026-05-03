import io
import json
import math
import os
import re
import shutil
import threading
import time
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
from tqdm.auto import tqdm
from yahooquery import Ticker

from mongo_store import get_mongo_store

warnings.filterwarnings("ignore")
load_dotenv()


@dataclass
class PipelineConfig:
    # This is basically the script's control panel. Most of the "why did it do that?"
    # questions trace back to one of these thresholds or cache TTLs.
    universe_history_years: int = 3
    universe_min_market_cap: float = 100e6
    universe_max_market_cap: float = 50e9
    screen_max_market_cap: float = 10e9
    lookback_months: int = 8
    min_return_pct: float = 30.0
    min_trading_days: int = 80
    min_last_price: float = 1.0
    min_recent_r2: float = 0.85
    weight_r2_ratio: float = 0.50
    weight_slope_ratio: float = 0.30
    weight_tnr_ratio: float = 0.20
    ratio_clip_min: float = 0.05
    ratio_clip_max: float = 10.0
    meta_chunk_size: int = 150
    meta_sleep_sec: float = 0.05
    meta_max_workers: int = 4
    metadata_refresh_days: int = 7
    download_chunk_size: int = 100
    price_sleep_sec: float = 0.02
    insider_lookback_days: int = 60
    insider_score_threshold: float = 25.0
    insider_max_workers: int = 8
    insider_timeout_sec: int = 12
    min_insider_coverage_ratio: float = 0.95
    package_max_workers: int = 6
    package_refresh_days: int = 60
    sec_min_interval_sec: float = 0.12
    sec_timeout_sec: int = 30
    sec_num_8k: int = 10

#folder paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
SCREENING_OUTPUT_DIR = BASE_DIR / "screening_output"
AGENTS_DATA_PACKAGE_DIR = BASE_DIR / "agents_data_package"
CHART_IMAGES_DIR = SCREENING_OUTPUT_DIR / "chart_images"
PRICE_FEATHER = DATA_DIR / "small_midcap_prices_3y.feather"
META_FEATHER = DATA_DIR / "small_midcap_meta.feather"
INSIDER_FEATHER = DATA_DIR / "insider_latest.feather"
INSIDER_SUMMARY_FEATHER = DATA_DIR / "insider_summary_latest.feather"

#splits a big list into chunks 
def chunk_list(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]

#scrapes the nasdaq website to the all the tickers on nasdaq and other exchanges
def download_nasdaq_symbol_dirs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    other_url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
    nasdaq_txt = requests.get(nasdaq_url, timeout=30).text
    other_txt = requests.get(other_url, timeout=30).text
    return pd.read_csv(io.StringIO(nasdaq_txt), sep="|"), pd.read_csv(io.StringIO(other_txt), sep="|")

#uses the previous funciton to get all the data, then builds a clean dataframe which is now the "stock universe"
def build_us_common_stock_universe() -> pd.DataFrame:
    nasdaq_df, other_df = download_nasdaq_symbol_dirs()
    nasdaq_df = nasdaq_df[nasdaq_df["Symbol"] != "File Creation Time"].copy()
    other_df = other_df[other_df["ACT Symbol"] != "File Creation Time"].copy()

    nasdaq = pd.DataFrame(
        {
            "Ticker": nasdaq_df["Symbol"].astype(str).str.upper().str.strip(),
            "Company": nasdaq_df["Security Name"].astype(str).str.strip(),
            "Exchange": "NASDAQ",
            "ETF": nasdaq_df.get("ETF", "N").astype(str).str.upper().eq("Y"),
            "TestIssue": nasdaq_df.get("Test Issue", "N").astype(str).str.upper().eq("Y"),
        }
    )
    exchange_map = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "BATS", "V": "IEX"}
    other = pd.DataFrame(
        {
            "Ticker": other_df["ACT Symbol"].astype(str).str.upper().str.strip(),
            "Company": other_df["Security Name"].astype(str).str.strip(),
            "Exchange": other_df["Exchange"].map(exchange_map).fillna(other_df["Exchange"].astype(str)),
            "ETF": other_df.get("ETF", "N").astype(str).str.upper().eq("Y"),
            "TestIssue": other_df.get("Test Issue", "N").astype(str).str.upper().eq("Y"),
        }
    )
    universe = pd.concat([nasdaq, other], ignore_index=True).drop_duplicates(subset=["Ticker"])
    universe = universe[(~universe["ETF"]) & (~universe["TestIssue"])].copy()

    # The raw exchange lists contain a lot of stuff that trades but is not a plain
    # common stock. This regex pass is the blunt cleanup layer.
    exclude_patterns = [
        r"\bETF\b",
        r"\bETN\b",
        r"\bFUND\b",
        r"\bTRUST\b",
        r"\bWARRANT\b",
        r"\bRIGHT\b",
        r"\bUNIT\b",
        r"\bPREFERRED\b",
        r"\bPREF\b",
        r"\bDEPOSITARY\b",
        r"\bNOTE\b",
        r"\bBOND\b",
        r"\bADR\b",
        r"\bADS\b",
        r"\bLP\b",
        r"\bL\.P\.\b",
        r"\bLIMITED PARTNERSHIP\b",
        r"\bACQUISITION\b",
        r"\bSPAC\b",
    ]
    pattern = "|".join(exclude_patterns)
    universe = universe[~universe["Company"].str.upper().str.contains(pattern, regex=True, na=False)].copy()
    universe = universe[~universe["Ticker"].str.contains(r"[\^\$]", regex=True, na=False)].copy()
    return universe.reset_index(drop=True)

#This function takes a list of tickers, and retrieves metadata of those tickers/stocks from Yahoo finnace
#(e.g. ticker, market_cap_num, sector, industry, etc.)
#splits the list into chunks and batches them and runs the reequests concurrently using threads so it is faster and also won't overwhelm the api/website.
#returns a pandas dataframe where each row is one ticker and each column is a piece of metadata. 
#The reason we need this function is because the Nasdaq symbol list only gives you symbols, not enough info to actually screen
def fetch_yahoo_metadata(
    tickers: List[str], chunk_size: int = 150, sleep_sec: float = 0.05, max_workers: int = 4
) -> pd.DataFrame:
    def _fetch_chunk(chunk_items: List[str], chunk_idx: int) -> List[dict]:
        print(f"Metadata chunk {chunk_idx}: {len(chunk_items)}")
        local_rows: List[dict] = []
        try:
            tq = Ticker(chunk_items, asynchronous=True, validate=True)
            price_data = tq.price
            profiles = tq.summary_profile
        except Exception as exc:
            print(f"  Metadata failure (chunk {chunk_idx}): {exc}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            return local_rows

        for ticker in chunk_items:
            p = price_data.get(ticker, {}) if isinstance(price_data, dict) else {}
            sp = profiles.get(ticker, {}) if isinstance(profiles, dict) else {}
            if not isinstance(p, dict):
                p = {}
            if not isinstance(sp, dict):
                sp = {}
            local_rows.append(
                {
                    "Ticker": ticker,
                    "market_cap_num": p.get("marketCap", np.nan),
                    "quoteType": p.get("quoteType"),
                    "exchangeName": p.get("exchangeName"),
                    "shortName": p.get("shortName"),
                    "regularMarketPrice": p.get("regularMarketPrice", np.nan),
                    "sector": sp.get("sector"),
                    "industry": sp.get("industry"),
                }
            )
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return local_rows

    rows: List[dict] = []
    chunks = list(chunk_list(tickers, chunk_size))
    worker_count = min(max_workers, max(1, len(chunks)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_fetch_chunk, chunk, idx) for idx, chunk in enumerate(chunks, start=1)]
        for future in as_completed(futures):
            rows.extend(future.result())

    meta = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"])
    if meta.empty:
        return meta
    meta["market_cap_num"] = pd.to_numeric(meta["market_cap_num"], errors="coerce")
    meta["regularMarketPrice"] = pd.to_numeric(meta["regularMarketPrice"], errors="coerce")
    return meta


def update_metadata_cache(raw_universe: pd.DataFrame, cfg: PipelineConfig, force_refresh: bool = False) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    tickers = raw_universe["Ticker"].dropna().astype(str).str.upper().tolist()
    cached = load_feather(META_FEATHER)

    if (not force_refresh) and cached is not None and not cached.empty and "meta_asof_date" in cached.columns:
        cached["meta_asof_date"] = pd.to_datetime(cached["meta_asof_date"], errors="coerce").dt.normalize()
        latest_asof = cached["meta_asof_date"].max()
        cached_tickers = set(cached["Ticker"].dropna().astype(str).str.upper())
        age_days = (today - latest_asof).days if pd.notna(latest_asof) else 99999
        missing = sorted(set(tickers) - cached_tickers)

        if age_days <= cfg.metadata_refresh_days and not missing:
            print(
                f"Metadata cache age {age_days} day(s) <= {cfg.metadata_refresh_days}; "
                "using cached metadata."
            )
            return cached[cached["Ticker"].isin(tickers)].drop_duplicates(subset=["Ticker"]).copy()

        if age_days <= cfg.metadata_refresh_days and missing:
            # Fresh cache, incomplete coverage. Fill only the holes instead of hammering Yahoo again.
            print(
                f"Metadata cache is fresh ({age_days} day(s)) but missing {len(missing)} ticker(s); "
                "fetching only missing metadata."
            )
            meta_missing = fetch_yahoo_metadata(
                missing,
                chunk_size=cfg.meta_chunk_size,
                sleep_sec=cfg.meta_sleep_sec,
                max_workers=cfg.meta_max_workers,
            )
            if not meta_missing.empty:
                meta_missing["meta_asof_date"] = today
                merged = pd.concat([cached, meta_missing], ignore_index=True)
                merged = merged.sort_values("meta_asof_date").drop_duplicates(subset=["Ticker"], keep="last")
                save_feather(merged, META_FEATHER)
                return merged[merged["Ticker"].isin(tickers)].drop_duplicates(subset=["Ticker"]).copy()
            return cached[cached["Ticker"].isin(tickers)].drop_duplicates(subset=["Ticker"]).copy()

    if force_refresh:
        print("Forced metadata refresh requested.")
    else:
        print(f"Metadata cache stale; refreshing full metadata for {len(tickers)} tickers.")

    meta = fetch_yahoo_metadata(
        tickers,
        chunk_size=cfg.meta_chunk_size,
        sleep_sec=cfg.meta_sleep_sec,
        max_workers=cfg.meta_max_workers,
    )
    meta["meta_asof_date"] = today
    save_feather(meta, META_FEATHER)
    return meta


def download_close_prices_long(
    tickers: List[str], start_date: str, end_date: str, chunk_size: int = 100, sleep_sec: float = 0.02
) -> pd.DataFrame:
    long_frames = []
    for i, chunk in enumerate(chunk_list(tickers, chunk_size), start=1):
        print(f"Price chunk {i}: {len(chunk)}")
        try:
            data = yf.download(
                tickers=chunk,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            print(f"  Price failure: {exc}")
            continue
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in chunk:
                if ticker not in data.columns.get_level_values(0):
                    continue
                if "Close" not in data[ticker]:
                    continue
                s = data[ticker]["Close"].dropna()
                if s.empty:
                    continue
                df = s.to_frame("close").reset_index().rename(columns={"Date": "date"})
                df["Ticker"] = ticker
                long_frames.append(df[["date", "Ticker", "close"]])
        else:
            if len(chunk) == 1 and "Close" in data.columns:
                s = data["Close"].dropna()
                if not s.empty:
                    df = s.to_frame("close").reset_index().rename(columns={"Date": "date"})
                    df["Ticker"] = chunk[0]
                    long_frames.append(df[["date", "Ticker", "close"]])
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if not long_frames:
        return pd.DataFrame(columns=["date", "Ticker", "close"])
    out = pd.concat(long_frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["close"])


def save_feather(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(path)


def load_feather(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_feather(path)


def get_latest_market_trading_day() -> pd.Timestamp:
    """
    Infer latest US market trading day from recent SPY bars.
    Falls back to previous business day on fetch issues.
    """
    today = pd.Timestamp.now().normalize()
    try:
        spy = yf.download(
            tickers="SPY",
            period="10d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if spy is not None and not spy.empty:
            idx = pd.to_datetime(spy.index).normalize()
            return pd.Timestamp(idx.max())
    except Exception:
        pass

    # Fallback if market probe fails.
    if today.weekday() >= 5:
        return today - pd.offsets.BDay(1)
    return today


def update_price_cache(universe_meta: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    today = pd.Timestamp.now().normalize()
    latest_market_day = get_latest_market_trading_day()
    cached = load_feather(PRICE_FEATHER)

    if cached is not None and not cached.empty:
        cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
        most_recent = cached["date"].max()
        print(f"Cache exists; latest cached date: {most_recent.date()} | latest market day: {latest_market_day.date()}")
        if most_recent >= latest_market_day:
            print("Cache is already up-to-date for the latest market trading day; using cached prices.")
            return cached
        # The +1 day on the end date is intentional: Yahoo's end date is effectively exclusive.
        fetch_start = (most_recent + timedelta(days=1)).strftime("%Y-%m-%d")
        print(
            f"Fetching incremental prices only from {fetch_start} to {(latest_market_day + timedelta(days=1)).date()}"
        )
    else:
        cached = pd.DataFrame(columns=["date", "Ticker", "close"])
        fetch_start = (today - pd.DateOffset(years=cfg.universe_history_years)).strftime("%Y-%m-%d")
        print(f"No cache found; fetching full history from {fetch_start}")

    tickers = sorted(universe_meta["Ticker"].dropna().unique().tolist())
    new_prices = download_close_prices_long(
        tickers=tickers,
        start_date=fetch_start,
        end_date=(latest_market_day + timedelta(days=1)).strftime("%Y-%m-%d"),
        chunk_size=cfg.download_chunk_size,
        sleep_sec=cfg.price_sleep_sec,
    )

    merged = pd.concat([cached, new_prices], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "Ticker"], keep="last")
    merged = merged.sort_values(["date", "Ticker"]).reset_index(drop=True)
    save_feather(merged, PRICE_FEATHER)
    return merged


def safe_ratio(recent: float, old: float, clip_min: float = 0.05, clip_max: float = 10.0) -> float:
    if pd.isna(recent) or pd.isna(old):
        return np.nan
    recent_adj = max(float(recent), clip_min)
    old_adj = max(float(old), clip_min)
    ratio = recent_adj / old_adj
    return min(max(ratio, 1 / clip_max), clip_max)


def trend_to_noise_ratio(price_series: pd.Series) -> float:
    s = pd.Series(price_series).dropna().copy()
    if len(s) < 3:
        return np.nan
    if s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan

    log_ret = np.log(s / s.shift(1)).dropna()
    if len(log_ret) == 0:
        return np.nan

    net_log_move = abs(np.log(s.iloc[-1] / s.iloc[0]))
    total_abs_log_move = np.abs(log_ret).sum()
    if total_abs_log_move == 0:
        return np.nan
    return float(net_log_move / total_abs_log_move)


def quadratic_log_fit_r2(price_series: pd.Series) -> Tuple[float, float]:
    s = pd.Series(price_series).dropna().copy()
    if len(s) < 5:
        return np.nan, np.nan
    if (s <= 0).any():
        return np.nan, np.nan

    y = np.log(s.values)
    t = np.arange(len(y))
    c_quad, b_quad, a_quad = np.polyfit(t, y, 2)
    y_hat = a_quad + b_quad * t + c_quad * (t ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return float(c_quad), float(r2)


def analyze_one_stock(series: pd.Series, cfg: PipelineConfig) -> Optional[dict]:
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna().copy()
    if len(s) < cfg.min_trading_days:
        return None
    if s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return None

    start_price = float(s.iloc[0])
    end_price = float(s.iloc[-1])
    total_return_pct = (end_price / start_price - 1) * 100
    if total_return_pct < cfg.min_return_pct:
        return None
    if end_price < cfg.min_last_price:
        return None

    # The script scores "is the trend getting cleaner and steeper lately?" by comparing
    # the older half of the chart to the recent half, not by fitting one model to the
    # whole period and hoping that tells the story.
    n = len(s)
    mid = n // 2
    old_s = s.iloc[:mid].copy()
    recent_s = s.iloc[mid:].copy()

    if len(old_s) < 20 or len(recent_s) < 20:
        return None
    if (old_s <= 0).any() or (recent_s <= 0).any():
        return None

    x_old = np.arange(len(old_s))
    y_old = np.log(old_s.to_numpy(dtype=float))
    reg_old = linregress(x_old, y_old)
    old_log_slope = float(reg_old.slope)
    old_r2 = float(reg_old.rvalue ** 2)
    old_tnr = trend_to_noise_ratio(old_s)

    x_recent = np.arange(len(recent_s))
    y_recent = np.log(recent_s.to_numpy(dtype=float))
    reg_recent = linregress(x_recent, y_recent)
    recent_log_slope = float(reg_recent.slope)
    recent_r2 = float(reg_recent.rvalue ** 2)
    recent_tnr = trend_to_noise_ratio(recent_s)

    if recent_r2 < cfg.min_recent_r2:
        return None
    if recent_log_slope <= 0:
        return None

    # Ratios are clipped on purpose so one weird denominator does not blow up the ranking.
    r2_ratio = safe_ratio(recent_r2, old_r2, cfg.ratio_clip_min, cfg.ratio_clip_max)
    slope_ratio = safe_ratio(recent_log_slope, max(old_log_slope, cfg.ratio_clip_min), cfg.ratio_clip_min, cfg.ratio_clip_max)
    tnr_ratio = safe_ratio(recent_tnr, old_tnr, cfg.ratio_clip_min, cfg.ratio_clip_max)

    technical_score = (
        cfg.weight_r2_ratio * r2_ratio
        + cfg.weight_slope_ratio * slope_ratio
        + cfg.weight_tnr_ratio * tnr_ratio
    )

    quadratic_c_curvature, quadratic_r2 = quadratic_log_fit_r2(s)

    return {
        "n_days": len(s),
        "start_price": start_price,
        "end_price": end_price,
        "total_return_pct": total_return_pct,
        "technical_score": float(technical_score),
        "recent_r2_old_r2_ratio": r2_ratio,
        "recent_r2": recent_r2,
        "old_r2": old_r2,
        "recent_tnr_old_tnr_ratio": tnr_ratio,
        "recent_tnr": recent_tnr,
        "old_tnr": old_tnr,
        "recent_log_slope": recent_log_slope,
        "old_log_slope": old_log_slope,
        "recent_log_slope_old_log_slope_ratio": slope_ratio,
        "quadratic_c_curvature": quadratic_c_curvature,
        "quadratic_r2": quadratic_r2,
    }


def _clean_colname(col: str) -> str:
    return str(col).strip().replace("\xa0", " ").replace("\n", " ").replace("\r", " ")


def _to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", "-", "nan", "None"}:
        return np.nan
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("+", "")
    s = s.replace("(", "-").replace(")", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def _extract_trade_code(val: str) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    return s.split(" - ")[0].strip().upper()


def clean_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def parse_openinsider_table_from_html(html: str, ticker: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return pd.DataFrame()
    if len(tables) == 0:
        return pd.DataFrame()

    df = max(tables, key=lambda x: x.shape[1]).copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in col if str(x) != "nan"]).strip() for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\xa0", " ")).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if "filing" in cl and "date" in cl:
            rename_map[c] = "filing_date"
        elif "trade" in cl and "date" in cl:
            rename_map[c] = "trade_date"
        elif cl == "ticker":
            rename_map[c] = "Ticker"
        elif "insider name" in cl or cl == "insider":
            rename_map[c] = "insider_name"
        elif "trade type" in cl or "trade code" in cl:
            rename_map[c] = "trade_code"
        elif cl == "title":
            rename_map[c] = "title"
        elif cl in {"price", "qty", "owned", "value"}:
            rename_map[c] = cl
    df = df.rename(columns=rename_map)

    if "Ticker" not in df.columns:
        df["Ticker"] = ticker
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df[df["Ticker"] == ticker.upper()].copy()

    for c in ["filing_date", "trade_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["value", "qty", "price", "owned"]:
        if c in df.columns:
            df[c] = clean_numeric_series(df[c])
    if "trade_code" in df.columns:
        df["trade_code"] = (
            df["trade_code"].astype(str).str.replace("\xa0", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.upper().str.strip()
        )
    if "insider_name" in df.columns:
        df["insider_name"] = (
            df["insider_name"].astype(str).str.replace("\xa0", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
        )
    return df


def fetch_openinsider_ticker_table(ticker: str, timeout: int = 20) -> pd.DataFrame:
    url = f"http://openinsider.com/search?q={ticker}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return parse_openinsider_table_from_html(resp.text, ticker=ticker)


def is_purchase_row(row: pd.Series) -> bool:
    if "trade_code" not in row.index or pd.isna(row["trade_code"]):
        return False
    return str(row["trade_code"]).upper().strip().startswith("P")


def summarize_insider(df: pd.DataFrame, lookback_days: int, market_cap: float) -> dict:
    base = {
        "buy_dollars_60d": 0.0,
        "unique_buyers_60d": 0,
        "insider_score_60d": 0.0,
        "n_insider_rows": 0,
    }
    if df is None or df.empty:
        return base
    date_col = "trade_date" if "trade_date" in df.columns else "filing_date"
    if date_col not in df.columns:
        return base
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    df = df[df[date_col] >= cutoff].copy()
    if df.empty:
        return base

    buys = df[df.apply(is_purchase_row, axis=1)].copy() if "trade_code" in df.columns else pd.DataFrame(columns=df.columns)
    if "value" in buys.columns:
        buys["value"] = pd.to_numeric(buys["value"], errors="coerce").fillna(0)
        buy_dollars = float(buys["value"].sum())
    else:
        buy_dollars = 0.0
    if "insider_name" in buys.columns:
        unique_buyers = int(buys["insider_name"].dropna().astype(str).str.strip().nunique())
    else:
        unique_buyers = 0
    # This is a homemade score, not a finance-standard metric. It rewards more dollars
    # and more distinct buyers, but uses log1p so one giant trade does not dominate everything.
    insider_score = math.log1p(max(buy_dollars, 0)) * (1 + 0.25 * unique_buyers)
    return {
        "buy_dollars_60d": buy_dollars,
        "unique_buyers_60d": unique_buyers,
        "insider_score_60d": float(insider_score),
        "n_insider_rows": int(len(df)),
    }


def build_screen(universe_meta: pd.DataFrame, price_long: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_start = pd.Timestamp.now().normalize() - pd.DateOffset(months=cfg.lookback_months)
    price_wide = (
        price_long.pivot(index="date", columns="Ticker", values="close")
        .sort_index()
        .loc[lambda x: x.index >= target_start]
    )
    screen_meta = universe_meta[
        (universe_meta["market_cap_num"] >= cfg.universe_min_market_cap)
        & (universe_meta["market_cap_num"] <= cfg.screen_max_market_cap)
    ].copy()
    tickers = sorted(set(price_wide.columns).intersection(set(screen_meta["Ticker"])))

    rows = []
    for ticker in tickers:
        stats = analyze_one_stock(price_wide[ticker], cfg)
        if stats is None:
            continue
        stats["Ticker"] = ticker
        rows.append(stats)
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked, price_wide
    ranked = ranked.merge(screen_meta[["Ticker", "Company", "sector", "industry", "market_cap_num"]], on="Ticker", how="left")
    ranked["score_r2"] = ranked["recent_r2"]
    return (
        ranked.sort_values(
            by=["technical_score", "recent_r2", "recent_log_slope", "total_return_pct"],
            ascending=False,
        ).reset_index(drop=True),
        price_wide,
    )


def add_insider_scores(ranked_df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ranked_df.empty:
        return ranked_df, pd.DataFrame()
    candidate_tickers = ranked_df["Ticker"].dropna().astype(str).str.upper().tolist()
    market_cap_map = (
        ranked_df[["Ticker", "market_cap_num"]]
        .dropna(subset=["Ticker"])
        .drop_duplicates(subset=["Ticker"])
        .set_index("Ticker")["market_cap_num"]
        .to_dict()
    )

    def _fetch_one(ticker: str):
        try:
            raw_local = fetch_openinsider_ticker_table(ticker=ticker, timeout=cfg.insider_timeout_sec)
        except Exception:
            raw_local = pd.DataFrame()
        summary_local = summarize_insider(
            raw_local,
            cfg.insider_lookback_days,
            float(market_cap_map.get(ticker, np.nan)) if pd.notna(market_cap_map.get(ticker, np.nan)) else np.nan,
        )
        summary_local["Ticker"] = ticker
        return raw_local, summary_local

    raw_rows = []
    summary_rows = []
    max_workers = max(1, int(cfg.insider_max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one, ticker) for ticker in candidate_tickers]
        for future in as_completed(futures):
            raw, summary = future.result()
            if not raw.empty:
                raw_rows.append(raw)
            summary_rows.append(summary)

    insider_raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    insider_summary_df = pd.DataFrame(summary_rows)
    merged = ranked_df.merge(insider_summary_df, on="Ticker", how="left")
    for c in ["buy_dollars_60d", "unique_buyers_60d", "insider_score_60d", "n_insider_rows"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
    merged["combined_score"] = (
        0.65 * merged["technical_score"].rank(pct=True).fillna(0)
        + 0.35 * merged["insider_score_60d"].rank(pct=True).fillna(0)
    )
    merged = merged.sort_values(["combined_score", "technical_score"], ascending=False).reset_index(drop=True)
    return merged, insider_raw_df


def build_insider_summary_for_tickers(
    candidate_tickers: List[str], cfg: PipelineConfig, market_cap_map: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    market_cap_map = market_cap_map or {}
    tickers = sorted({str(t).strip().upper() for t in candidate_tickers if str(t).strip()})

    def _fetch_one(ticker: str):
        try:
            raw_local = fetch_openinsider_ticker_table(ticker=ticker, timeout=cfg.insider_timeout_sec)
        except Exception:
            raw_local = pd.DataFrame()
        cap = market_cap_map.get(ticker, np.nan)
        summary_local = summarize_insider(
            raw_local,
            cfg.insider_lookback_days,
            float(cap) if pd.notna(cap) else np.nan,
        )
        summary_local["Ticker"] = ticker
        return raw_local, summary_local

    raw_rows = []
    summary_rows = []
    max_workers = max(1, int(cfg.insider_max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one, ticker) for ticker in tickers]
        with tqdm(total=len(futures), desc="Insider scrape", unit="ticker") as pbar:
            for future in as_completed(futures):
                raw, summary = future.result()
                if not raw.empty:
                    raw_rows.append(raw)
                summary_rows.append(summary)
                pbar.update(1)

    insider_raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    insider_summary_df = pd.DataFrame(summary_rows)
    if insider_summary_df.empty:
        insider_summary_df = pd.DataFrame({"Ticker": tickers})
    for c in [
        "buy_dollars_60d",
        "unique_buyers_60d",
        "insider_score_60d",
        "n_insider_rows",
    ]:
        if c not in insider_summary_df.columns:
            insider_summary_df[c] = 0.0
        insider_summary_df[c] = insider_summary_df[c].fillna(0)
    insider_summary_df = insider_summary_df.sort_values(
        by=["insider_score_60d", "buy_dollars_60d", "unique_buyers_60d"], ascending=False
    ).reset_index(drop=True)
    return insider_summary_df, insider_raw_df


def _dedupe_insider_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Ticker" in out.columns:
        out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    dedupe_cols = [c for c in ["Ticker", "trade_date", "insider_name", "trade_type", "price", "qty", "value"] if c in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        out = out.drop_duplicates(keep="last")
    return out.reset_index(drop=True)


def update_insider_cache(
    candidate_tickers: List[str], cfg: PipelineConfig, market_cap_map: Optional[dict] = None, force_refresh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest_market_day = get_latest_market_trading_day()
    tickers = sorted({str(t).strip().upper() for t in candidate_tickers if str(t).strip()})
    market_cap_map = market_cap_map or {}

    cached_summary = load_feather(INSIDER_SUMMARY_FEATHER)
    cached_raw = load_feather(INSIDER_FEATHER)

    if not force_refresh and cached_summary is not None and not cached_summary.empty and "insider_asof_date" in cached_summary.columns:
        cached_summary["insider_asof_date"] = pd.to_datetime(cached_summary["insider_asof_date"], errors="coerce").dt.normalize()
        latest_asof = cached_summary["insider_asof_date"].max()
        cached_tickers = set(cached_summary["Ticker"].dropna().astype(str).str.upper())
        missing = sorted(set(tickers) - cached_tickers)

        if pd.notna(latest_asof) and latest_asof >= latest_market_day and not missing:
            print(
                f"Insider cache up-to-date (asof {latest_asof.date()}, market day {latest_market_day.date()}); using cache."
            )
            summary = cached_summary[cached_summary["Ticker"].isin(tickers)].drop_duplicates(subset=["Ticker"]).copy()
            raw = pd.DataFrame()
            if cached_raw is not None and not cached_raw.empty and "Ticker" in cached_raw.columns:
                raw = cached_raw[cached_raw["Ticker"].astype(str).str.upper().isin(tickers)].copy()
            return summary, raw

        if pd.notna(latest_asof) and latest_asof >= latest_market_day and missing:
            # Same idea as metadata: if today's cache is good but incomplete, only scrape the missing names.
            print(f"Insider cache up-to-date but missing {len(missing)} ticker(s); fetching only missing.")
            fresh_summary, fresh_raw = build_insider_summary_for_tickers(missing, cfg, market_cap_map)
            fresh_summary["insider_asof_date"] = latest_market_day
            merged_summary = pd.concat([cached_summary, fresh_summary], ignore_index=True)
            merged_summary = merged_summary.sort_values("insider_asof_date").drop_duplicates(subset=["Ticker"], keep="last")

            merged_raw = pd.DataFrame() if cached_raw is None else cached_raw.copy()
            if fresh_raw is not None and not fresh_raw.empty:
                merged_raw = _dedupe_insider_raw(pd.concat([merged_raw, fresh_raw], ignore_index=True))

            save_feather(merged_summary, INSIDER_SUMMARY_FEATHER)
            save_feather(merged_raw, INSIDER_FEATHER)
            out_summary = merged_summary[merged_summary["Ticker"].isin(tickers)].copy()
            out_raw = pd.DataFrame()
            if not merged_raw.empty and "Ticker" in merged_raw.columns:
                out_raw = merged_raw[merged_raw["Ticker"].astype(str).str.upper().isin(tickers)].copy()
            return out_summary, out_raw

    if force_refresh:
        print("Forced insider refresh requested.")
    else:
        print("Insider cache stale or missing; fetching and merging latest insider data.")

    fresh_summary, fresh_raw = build_insider_summary_for_tickers(tickers, cfg, market_cap_map)
    fresh_summary["insider_asof_date"] = latest_market_day

    merged_summary = fresh_summary.copy()
    if cached_summary is not None and not cached_summary.empty:
        merged_summary = pd.concat([cached_summary, fresh_summary], ignore_index=True)
        if "insider_asof_date" in merged_summary.columns:
            merged_summary["insider_asof_date"] = pd.to_datetime(
                merged_summary["insider_asof_date"], errors="coerce"
            ).dt.normalize()
            merged_summary = merged_summary.sort_values("insider_asof_date")
        merged_summary = merged_summary.drop_duplicates(subset=["Ticker"], keep="last")
    merged_summary = merged_summary.reset_index(drop=True)

    merged_raw = _dedupe_insider_raw(fresh_raw)
    if cached_raw is not None and not cached_raw.empty:
        merged_raw = _dedupe_insider_raw(pd.concat([cached_raw, fresh_raw], ignore_index=True))

    save_feather(merged_summary, INSIDER_SUMMARY_FEATHER)
    save_feather(merged_raw, INSIDER_FEATHER)
    out_summary = merged_summary[merged_summary["Ticker"].isin(tickers)].copy()
    out_raw = pd.DataFrame()
    if not merged_raw.empty and "Ticker" in merged_raw.columns:
        out_raw = merged_raw[merged_raw["Ticker"].astype(str).str.upper().isin(tickers)].copy()
    return out_summary, out_raw


def build_screening_union_df(
    technical_df: pd.DataFrame, insider_ranked_df: pd.DataFrame, universe_meta: pd.DataFrame, insider_threshold: float
) -> pd.DataFrame:
    if technical_df is None or technical_df.empty or "Ticker" not in technical_df.columns:
        technical_enriched = pd.DataFrame(columns=["Ticker"])
    else:
        already_has = {"buy_dollars_60d", "unique_buyers_60d", "insider_score_60d", "n_insider_rows"}.issubset(
            set(technical_df.columns)
        )
        if already_has:
            technical_enriched = technical_df.copy()
        else:
            technical_enriched = technical_df.merge(
                insider_ranked_df[
                    [
                        "Ticker",
                        "buy_dollars_60d",
                        "unique_buyers_60d",
                        "insider_score_60d",
                        "n_insider_rows",
                    ]
                ],
                on="Ticker",
                how="left",
            )
    insider_only = insider_ranked_df[insider_ranked_df["insider_score_60d"] > insider_threshold].copy()
    insider_only = insider_only.merge(
        universe_meta[["Ticker", "Company", "sector", "industry", "market_cap_num"]],
        on="Ticker",
        how="left",
    )

    final_df = pd.concat([technical_enriched, insider_only], ignore_index=True, sort=False)
    # Final list is the union of:
    # 1) names that passed the technical screen
    # 2) names with unusually strong insider buying, even if they missed the chart screen
    final_df["from_technical_screen"] = final_df["technical_score"].notna() if "technical_score" in final_df.columns else False
    final_df["from_insider_threshold"] = final_df["insider_score_60d"].fillna(0) > insider_threshold
    final_df = final_df.sort_values(
        by=["from_technical_screen", "insider_score_60d", "technical_score", "total_return_pct"],
        ascending=False,
    ).drop_duplicates(subset=["Ticker"], keep="first")
    final_df = final_df.reset_index(drop=True)
    return final_df


def merge_insider_data(base_df: pd.DataFrame, insider_summary_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.merge(insider_summary_df, on="Ticker", how="left")
    for c in ["buy_dollars_60d", "unique_buyers_60d", "insider_score_60d", "n_insider_rows"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
    return merged


def add_normalized_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["technical_score_norm"] = out["technical_score"].rank(pct=True)
    out["insider_score_norm"] = out["insider_score_60d"].rank(pct=True)
    out["combined_score"] = 0.65 * out["technical_score_norm"] + 0.35 * out["insider_score_norm"]
    return out


def validate_insider_coverage(df: pd.DataFrame, min_ratio: float) -> None:
    if df is None or df.empty:
        raise RuntimeError("Combined dataframe is empty; cannot validate insider coverage.")
    if "insider_score_60d" not in df.columns:
        raise RuntimeError("Missing insider_score_60d column in combined dataframe.")
    total = len(df)
    covered = int(df["insider_score_60d"].notna().sum())
    ratio = covered / total if total > 0 else 0
    if ratio < min_ratio:
        raise RuntimeError(
            f"Insider coverage too low: {covered}/{total} ({ratio:.1%}) < required {min_ratio:.1%}. "
            "This indicates insider fetch/cache issue; refusing to continue."
        )


def rank_screening_df(final_df: pd.DataFrame) -> pd.DataFrame:
    out = final_df.copy()
    sort_cols = [c for c in ["combined_score", "technical_score", "insider_score_60d", "total_return_pct"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    out["screen_rank"] = np.arange(1, len(out) + 1)
    return out


def select_rank_range(final_df: pd.DataFrame, start_rank: int = 1, end_rank: Optional[int] = None) -> pd.DataFrame:
    ranked = final_df.copy()
    if "screen_rank" not in ranked.columns:
        ranked = rank_screening_df(ranked)
    start_rank = max(1, int(start_rank))
    end_rank = int(end_rank) if end_rank is not None else int(ranked["screen_rank"].max())
    end_rank = max(start_rank, end_rank)
    return ranked[(ranked["screen_rank"] >= start_rank) & (ranked["screen_rank"] <= end_rank)].copy()


def resolve_rank_bounds(total_count: int, start_rank: int, end_rank: Optional[int], label: str) -> Tuple[int, int]:
    if total_count <= 0:
        raise ValueError(f"No screened stocks are available for {label}.")

    normalized_start = max(1, int(start_rank))
    if normalized_start != int(start_rank):
        print(f"{label}: adjusted start rank from {start_rank} to {normalized_start}.")

    if normalized_start > total_count:
        raise ValueError(
            f"{label}: start rank {normalized_start} exceeds available screened stocks ({total_count})."
        )

    normalized_end = total_count if end_rank is None else max(normalized_start, int(end_rank))
    if normalized_end > total_count:
        print(f"{label}: adjusted end rank from {normalized_end} to {total_count}.")
        normalized_end = total_count

    return normalized_start, normalized_end


def _build_chart_figure(row: pd.Series, price_wide: pd.DataFrame, cfg: PipelineConfig):
    target_start = pd.Timestamp.now().normalize() - pd.DateOffset(months=cfg.lookback_months)
    ticker = str(row["Ticker"]).upper()
    if ticker not in price_wide.columns:
        return None
    s = pd.to_numeric(price_wide[ticker], errors="coerce").dropna()
    s = s[s.index >= target_start]
    if len(s) < max(20, cfg.min_trading_days):
        return None
    if (s <= 0).any():
        return None

    n = len(s)
    mid = n // 2
    old_s = s.iloc[:mid].copy()
    recent_s = s.iloc[mid:].copy()
    if len(old_s) < 5 or len(recent_s) < 5:
        return None

    log_s = np.log(s.to_numpy(dtype=float))

    x_old = np.arange(len(old_s))
    y_old = np.log(old_s.values)
    reg_old = linregress(x_old, y_old)
    yhat_old = reg_old.intercept + reg_old.slope * x_old

    x_recent = np.arange(len(recent_s))
    y_recent = np.log(recent_s.values)
    reg_recent = linregress(x_recent, y_recent)
    yhat_recent = reg_recent.intercept + reg_recent.slope * x_recent

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(s.index, log_s, label="Actual log-price", linewidth=2)
    ax.plot(old_s.index, yhat_old, label=f"Old-half fit | slope={reg_old.slope:.5f}, R2={reg_old.rvalue**2:.3f}", linewidth=2)
    ax.plot(recent_s.index, yhat_recent, label=f"Recent-half fit | slope={reg_recent.slope:.5f}, R2={reg_recent.rvalue**2:.3f}", linewidth=2)
    ax.axvline(s.index[mid], linestyle="--", alpha=0.7, label="Split point")
    ax.grid(True, alpha=0.3)
    insider_score = row.get("insider_score_60d", np.nan)
    buy_dollars = row.get("buy_dollars_60d", np.nan)
    unique_buyers = row.get("unique_buyers_60d", np.nan)
    insider_txt = "MISSING" if pd.isna(insider_score) else f"{float(insider_score):.2f}"
    buy_txt = "MISSING" if pd.isna(buy_dollars) else f"{float(buy_dollars):,.0f}"
    ub_txt = "MISSING" if pd.isna(unique_buyers) else f"{int(unique_buyers)}"
    title = (
        f"{ticker} | Rank={int(row.get('screen_rank', -1))} | Combined={row.get('combined_score', np.nan):.3f} | "
        f"Tech={row.get('technical_score', np.nan):.3f} | "
        f"Insider={insider_txt} | Buy$60d={buy_txt} | UniqueBuyers={ub_txt}"
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Price")
    ax.legend()
    fig.tight_layout()
    return fig


def build_chart_images_for_screening(
    final_df: pd.DataFrame, price_wide: pd.DataFrame, cfg: PipelineConfig, output_dir: Path
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ranked = rank_screening_df(final_df)
    for _, row in ranked.iterrows():
        fig = _build_chart_figure(row, price_wide, cfg)
        if fig is None:
            continue
        ticker = str(row["Ticker"]).upper()
        image_path = output_dir / f"{int(row['screen_rank']):03d}_{ticker}.png"
        fig.savefig(image_path, dpi=160)
        plt.close(fig)
        rows.append(
            {
                "Ticker": ticker,
                "screen_rank": int(row["screen_rank"]),
                "chart_image_path": str(image_path),
            }
        )
    return pd.DataFrame(rows)


def build_chart_pdf_for_screening(
    final_df: pd.DataFrame,
    price_wide: pd.DataFrame,
    cfg: PipelineConfig,
    output_pdf: Path,
    start_rank: int = 1,
    end_rank: Optional[int] = None,
) -> pd.DataFrame:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    ranked = select_rank_range(final_df, start_rank=start_rank, end_rank=end_rank)
    included_rows = []
    with PdfPages(output_pdf) as pdf:
        for _, row in ranked.iterrows():
            fig = _build_chart_figure(row, price_wide, cfg)
            if fig is None:
                continue
            included_rows.append({"Ticker": str(row["Ticker"]).upper(), "screen_rank": int(row["screen_rank"])})
            pdf.savefig(fig)
            plt.close(fig)
    return pd.DataFrame(included_rows)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))


def _json_safe(val):
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(val))
    if pd.isna(val):
        return None
    return val


def _sec_headers() -> dict:
    user_agent = os.getenv("SEC_USER_AGENT", "StocksPipeline research your_email@example.com")
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _throttled_get(url: str, timeout: int, limiter: dict, headers: Optional[dict] = None) -> requests.Response:
    headers = headers or {}
    with limiter["lock"]:
        now = time.time()
        wait = limiter["next_ts"] - now
        if wait > 0:
            time.sleep(wait)
        limiter["next_ts"] = time.time() + limiter["interval"]
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp


def _select_recent_filings(df_recent: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if df_recent.empty:
        return df_recent
    df = df_recent.copy()
    df["form_upper"] = df["form"].astype(str).str.upper()
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    k = df[df["form_upper"] == "10-K"].sort_values("filingDate", ascending=False).head(1)
    q = df[df["form_upper"] == "10-Q"].sort_values("filingDate", ascending=False).head(2)
    e = df[df["form_upper"] == "8-K"].sort_values("filingDate", ascending=False).head(cfg.sec_num_8k)
    out = pd.concat([k, q, e], ignore_index=True)
    return out.drop_duplicates(subset=["accessionNumber"])


def _build_sec_doc_url(cik_10: str, accession_number: str, primary_document: str) -> str:
    accession_nodashes = accession_number.replace("-", "")
    cik_no_leading_zeros = str(int(cik_10))
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{accession_nodashes}/{primary_document}"


def build_agents_data_packages(
    final_df: pd.DataFrame, price_wide: pd.DataFrame, universe_meta: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    AGENTS_DATA_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
    sec_headers = _sec_headers()
    limiter = {"lock": threading.Lock(), "next_ts": 0.0, "interval": cfg.sec_min_interval_sec}

    try:
        ticker_map_resp = _throttled_get(ticker_map_url, cfg.sec_timeout_sec, limiter, headers=sec_headers).json()
        sec_ticker_to_cik = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in ticker_map_resp.values()}
    except Exception:
        sec_ticker_to_cik = {}

    expected_tickers = {str(t).upper() for t in final_df["Ticker"].dropna().astype(str)}
    for child in AGENTS_DATA_PACKAGE_DIR.iterdir():
        if child.is_dir() and child.name.upper() not in expected_tickers:
            shutil.rmtree(child, ignore_errors=True)

    meta_map = (
        universe_meta[["Ticker", "Company", "sector", "industry", "market_cap_num"]]
        .drop_duplicates(subset=["Ticker"])
        .set_index("Ticker")
        .to_dict(orient="index")
    )

    wanted_fields = [
        "shortName",
        "longName",
        "sector",
        "industry",
        "marketCap",
        "enterpriseValue",
        "sharesOutstanding",
        "floatShares",
        "currentPrice",
        "previousClose",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "averageVolume",
        "averageVolume10days",
        "beta",
        "trailingPE",
        "priceToBook",
        "totalCash",
        "totalDebt",
        "revenueGrowth",
        "grossMargins",
        "operatingMargins",
        "ebitdaMargins",
    ]

    def _save_df(df_obj: pd.DataFrame, csv_path: Path) -> None:
        if df_obj is None or df_obj.empty:
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_obj.to_csv(csv_path, index=True)

    def _to_ts(val) -> Optional[pd.Timestamp]:
        if val is None:
            return None
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts).normalize()

    today = pd.Timestamp.now().normalize()

    def _build_one(row: pd.Series) -> dict:
        ticker = str(row["Ticker"]).upper()
        pkg_dir = AGENTS_DATA_PACKAGE_DIR / _safe_filename(ticker)
        yahoo_dir = pkg_dir / "yahoo"
        sec_dir = pkg_dir / "sec"
        filings_dir = sec_dir / "filings_html"
        package_meta_path = pkg_dir / "package_meta.json"
        for d in [pkg_dir, yahoo_dir, sec_dir, filings_dir]:
            d.mkdir(parents=True, exist_ok=True)

        out = {"Ticker": ticker, "package_dir": str(pkg_dir), "package_status": "ok", "package_error": "", "cache_hit": False}
        try:
            existing_meta = {}
            if package_meta_path.exists():
                try:
                    existing_meta = json.loads(package_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_meta = {}

            package_asof = _to_ts(existing_meta.get("package_asof_date"))
            package_age_days = (today - package_asof).days if package_asof is not None else 99999
            is_package_fresh = package_age_days <= cfg.package_refresh_days

            cik_10 = sec_ticker_to_cik.get(ticker)
            out["cik_10"] = cik_10

            sec_has_new = True
            latest_selected_df = pd.DataFrame()
            latest_sec_filing_date = None
            if cik_10:
                # Cheap freshness check first: if SEC has not posted anything newer than what
                # we already packaged, we can skip rebuilding the whole folder.
                try:
                    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
                    submissions = _throttled_get(submissions_url, cfg.sec_timeout_sec, limiter, headers=sec_headers).json()
                    recent = submissions.get("filings", {}).get("recent", {})
                    df_recent = pd.DataFrame(recent) if isinstance(recent, dict) else pd.DataFrame()
                    latest_selected_df = _select_recent_filings(df_recent, cfg) if not df_recent.empty else pd.DataFrame()
                    if not latest_selected_df.empty and "filingDate" in latest_selected_df.columns:
                        latest_sec_filing_date = pd.to_datetime(latest_selected_df["filingDate"], errors="coerce").max()
                        latest_sec_filing_date = None if pd.isna(latest_sec_filing_date) else pd.Timestamp(latest_sec_filing_date).normalize()
                    prev_last = _to_ts(existing_meta.get("last_sec_filing_date"))
                    sec_has_new = True if prev_last is None or latest_sec_filing_date is None else (latest_sec_filing_date > prev_last)
                except Exception:
                    sec_has_new = False if is_package_fresh else True

            if is_package_fresh and (not sec_has_new):
                out["package_status"] = "cached"
                out["cache_hit"] = True
                return out

            summary = {k: _json_safe(v) for k, v in row.to_dict().items()}
            summary.update(meta_map.get(ticker, {}))
            with open(pkg_dir / "screening_snapshot.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

            if ticker in price_wide.columns:
                s = price_wide[ticker].dropna().to_frame("close").reset_index()
                s.columns = ["date", "close"]
                s.to_feather(pkg_dir / "price_history.feather")
                s.to_csv(pkg_dir / "price_history.csv", index=False)

            tk = yf.Ticker(ticker)
            fast_info = {}
            try:
                fi = getattr(tk, "fast_info", {})
                fast_info = dict(fi) if fi is not None else {}
            except Exception:
                fast_info = {}
            with open(yahoo_dir / "fast_info.json", "w", encoding="utf-8") as f:
                json.dump({k: _json_safe(v) for k, v in fast_info.items()}, f, indent=2, default=str)

            info = {}
            try:
                info = tk.info or {}
            except Exception:
                info = {}
            info_subset = {k: _json_safe(info.get(k, None)) for k in wanted_fields}
            with open(yahoo_dir / "info_selected.json", "w", encoding="utf-8") as f:
                json.dump(info_subset, f, indent=2, default=str)

            _save_df(getattr(tk, "financials", pd.DataFrame()), yahoo_dir / "financials.csv")
            _save_df(getattr(tk, "balance_sheet", pd.DataFrame()), yahoo_dir / "balance_sheet.csv")
            _save_df(getattr(tk, "cashflow", pd.DataFrame()), yahoo_dir / "cashflow.csv")
            _save_df(getattr(tk, "quarterly_financials", pd.DataFrame()), yahoo_dir / "quarterly_financials.csv")
            _save_df(getattr(tk, "quarterly_balance_sheet", pd.DataFrame()), yahoo_dir / "quarterly_balance_sheet.csv")
            _save_df(getattr(tk, "quarterly_cashflow", pd.DataFrame()), yahoo_dir / "quarterly_cashflow.csv")

            if cik_10:
                if latest_selected_df.empty:
                    try:
                        submissions_url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
                        submissions = _throttled_get(submissions_url, cfg.sec_timeout_sec, limiter, headers=sec_headers).json()
                        recent = submissions.get("filings", {}).get("recent", {})
                        df_recent = pd.DataFrame(recent) if isinstance(recent, dict) else pd.DataFrame()
                        latest_selected_df = _select_recent_filings(df_recent, cfg) if not df_recent.empty else pd.DataFrame()
                    except Exception:
                        latest_selected_df = pd.DataFrame()
                if not latest_selected_df.empty:
                    latest_selected_df.to_csv(sec_dir / "selected_filings.csv", index=False)
                    if filings_dir.exists():
                        shutil.rmtree(filings_dir, ignore_errors=True)
                    filings_dir.mkdir(parents=True, exist_ok=True)
                    for _, frow in latest_selected_df.iterrows():
                        acc = str(frow.get("accessionNumber", "")).strip()
                        doc = str(frow.get("primaryDocument", "")).strip()
                        if not acc or not doc:
                            continue
                        doc_url = _build_sec_doc_url(cik_10, acc, doc)
                        try:
                            content = _throttled_get(doc_url, 60, limiter, headers=sec_headers).content
                            save_name = _safe_filename(f"{acc}_{doc}")
                            with open(filings_dir / save_name, "wb") as fw:
                                fw.write(content)
                        except Exception:
                            continue

            latest_10k = None
            latest_10q = None
            latest_8k = None
            if not latest_selected_df.empty and "form" in latest_selected_df.columns and "filingDate" in latest_selected_df.columns:
                tmp = latest_selected_df.copy()
                tmp["form"] = tmp["form"].astype(str).str.upper()
                tmp["filingDate"] = pd.to_datetime(tmp["filingDate"], errors="coerce")
                if not tmp[tmp["form"] == "10-K"].empty:
                    latest_10k = str(tmp[tmp["form"] == "10-K"]["filingDate"].max().date())
                if not tmp[tmp["form"] == "10-Q"].empty:
                    latest_10q = str(tmp[tmp["form"] == "10-Q"]["filingDate"].max().date())
                if not tmp[tmp["form"] == "8-K"].empty:
                    latest_8k = str(tmp[tmp["form"] == "8-K"]["filingDate"].max().date())

            package_meta = {
                "ticker": ticker,
                "package_asof_date": str(today.date()),
                "package_age_days": 0,
                "cik_10": cik_10,
                "last_sec_filing_date": str(latest_sec_filing_date.date()) if latest_sec_filing_date is not None else existing_meta.get("last_sec_filing_date"),
                "last_10k_date": latest_10k,
                "last_10q_date": latest_10q,
                "last_8k_date": latest_8k,
                "refreshed_due_to_new_sec": bool(sec_has_new),
            }
            package_meta_path.write_text(json.dumps(package_meta, indent=2), encoding="utf-8")
            return out
        except Exception as exc:
            out["package_status"] = "error"
            out["package_error"] = str(exc)
            return out

    results = []
    worker_count = min(max(1, cfg.package_max_workers), max(1, len(final_df)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_build_one, row) for _, row in final_df.iterrows()]
        with tqdm(total=len(futures), desc="Building agent data packages", unit="ticker") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    return pd.DataFrame(results).sort_values("Ticker").reset_index(drop=True)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DATA_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock screening and data package pipeline")
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Force full metadata refresh from Yahoo (ignore metadata cache age).",
    )
    parser.add_argument(
        "--metadata-refresh-days",
        type=int,
        default=None,
        help="Override metadata cache TTL in days (default: 7).",
    )
    parser.add_argument(
        "--refresh-insider",
        action="store_true",
        help="Force insider cache refresh (ignore insider cache freshness).",
    )
    parser.add_argument(
        "--chart-start-rank",
        type=int,
        default=1,
        help="Start rank for the screening chart PDF bundle.",
    )
    parser.add_argument(
        "--chart-end-rank",
        type=int,
        default=50,
        help="End rank for the screening chart PDF bundle.",
    )
    parser.add_argument(
        "--package-start-rank",
        type=int,
        default=1,
        help="Start rank for building per-ticker agent data packages.",
    )
    parser.add_argument(
        "--package-end-rank",
        type=int,
        default=None,
        help="End rank for building per-ticker agent data packages. Defaults to all screened names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()
    if args.metadata_refresh_days is not None and args.metadata_refresh_days >= 0:
        cfg.metadata_refresh_days = int(args.metadata_refresh_days)
    ensure_dirs()

    print("1) Building universe and metadata...")
    raw_universe = build_us_common_stock_universe()
    meta = update_metadata_cache(raw_universe, cfg, force_refresh=bool(args.refresh_metadata))
    universe_meta = raw_universe.merge(meta, on="Ticker", how="left")
    universe_meta = universe_meta[universe_meta["quoteType"] == "EQUITY"].copy()
    universe_meta = universe_meta[
        (universe_meta["market_cap_num"] >= cfg.universe_min_market_cap)
        & (universe_meta["market_cap_num"] <= cfg.universe_max_market_cap)
    ].copy()
    universe_meta = universe_meta.reset_index(drop=True)
    print(f"Universe size (100M-50B): {len(universe_meta)}")

    print("2) Refreshing cached price data...")
    price_long = update_price_cache(universe_meta, cfg)
    print(f"Price rows: {len(price_long)}")
    price_wide = price_long.pivot(index="date", columns="Ticker", values="close").sort_index()

    print("3) Running insider pipeline first (all candidate tickers)...")
    # Insider is run on the broad candidate set before the technical filter so the final
    # union can still include names that are interesting only because of insider activity.
    candidate_tickers = sorted(set(price_wide.columns).intersection(set(universe_meta["Ticker"])))
    market_cap_map = universe_meta.set_index("Ticker")["market_cap_num"].to_dict()
    insider_ranked_df, insider_raw = update_insider_cache(
        candidate_tickers,
        cfg,
        market_cap_map=market_cap_map,
        force_refresh=bool(args.refresh_insider),
    )
    print(f"Insider summary rows: {len(insider_ranked_df)}")

    print("4) Running technical screen second...")
    technical_df, _ = build_screen(universe_meta, price_long, cfg)
    print(f"Technical pass rows: {len(technical_df)}")

    print("5) Building combined ranking (technical + insider)...")
    combined_df = merge_insider_data(technical_df, insider_ranked_df)
    # If insider coverage is missing for too many rows, the combined rank becomes misleading,
    # so the script stops instead of quietly pretending the data is complete.
    validate_insider_coverage(combined_df, cfg.min_insider_coverage_ratio)
    combined_df = add_normalized_scores(combined_df)
    combined_df = combined_df.sort_values(
        by=["combined_score", "technical_score", "recent_r2", "recent_log_slope"],
        ascending=False,
    ).reset_index(drop=True)

    print("6) Building final deduped screening union...")
    final_df = build_screening_union_df(
        technical_df=combined_df,
        insider_ranked_df=insider_ranked_df,
        universe_meta=universe_meta,
        insider_threshold=cfg.insider_score_threshold,
    )
    final_df = rank_screening_df(final_df)
    chart_start_rank, chart_end_rank = resolve_rank_bounds(
        len(final_df),
        args.chart_start_rank,
        args.chart_end_rank,
        label="Chart range",
    )
    package_start_rank, package_end_rank = resolve_rank_bounds(
        len(final_df),
        args.package_start_rank,
        args.package_end_rank,
        label="Package range",
    )
    package_df = select_rank_range(final_df, start_rank=package_start_rank, end_rank=package_end_rank)

    insider_ranked_path = SCREENING_OUTPUT_DIR / "insider_ranked.csv"
    combined_ranked_path = SCREENING_OUTPUT_DIR / "combined_ranked.csv"
    feather_path = SCREENING_OUTPUT_DIR / "final_screening_union.feather"
    csv_path = SCREENING_OUTPUT_DIR / "final_screening_union.csv"
    pdf_path = SCREENING_OUTPUT_DIR / "final_screening_charts.pdf"
    chart_manifest_path = SCREENING_OUTPUT_DIR / "chart_manifest.csv"

    insider_ranked_df.to_csv(insider_ranked_path, index=False)
    combined_df.to_csv(combined_ranked_path, index=False)
    save_feather(final_df, feather_path)
    final_df.to_csv(csv_path, index=False)

    print("7) Generating chart artifacts for screening names...")
    chart_manifest_df = build_chart_images_for_screening(final_df, price_wide, cfg, CHART_IMAGES_DIR)
    chart_manifest_df.to_csv(chart_manifest_path, index=False)
    selected_chart_rows = build_chart_pdf_for_screening(
        final_df,
        price_wide,
        cfg,
        pdf_path,
        start_rank=chart_start_rank,
        end_rank=chart_end_rank,
    )

    print("8) Building per-ticker agent data packages (Yahoo + SEC)...")
    package_manifest_df = build_agents_data_packages(package_df, price_wide, universe_meta, cfg)
    package_manifest_path = SCREENING_OUTPUT_DIR / "agents_data_package_manifest.csv"
    package_manifest_df.to_csv(package_manifest_path, index=False)

    mongo = get_mongo_store()
    screening_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mongo is not None:
        mongo.upsert_global_cache(
            "latest_screening_snapshot",
            {
                "screening_run_id": screening_run_id,
                "universe_size": int(len(universe_meta)),
                "technical_rows": int(len(technical_df)),
                "final_rows": int(len(final_df)),
                "latest_market_day": str(get_latest_market_trading_day().date()),
                "final_ranked_stocks": final_df.to_dict(orient="records"),
                "chart_manifest": chart_manifest_df.to_dict(orient="records"),
                "artifact_paths": {
                    "final_feather": str(feather_path),
                    "final_csv": str(csv_path),
                    "chart_manifest_csv": str(chart_manifest_path),
                    "screening_pdf": str(pdf_path),
                    "package_manifest_csv": str(package_manifest_path),
                },
                "selected_package_tickers": package_df[["Ticker", "screen_rank"]].to_dict(orient="records"),
            },
        )
        mongo.upsert_screening_run(
            screening_run_id,
            {
                "created_at": datetime.utcnow().isoformat(),
                "config": vars(cfg),
                "chart_range": {
                    "start_rank": int(chart_start_rank),
                    "end_rank": int(chart_end_rank),
                },
                "package_range": {
                    "start_rank": int(package_start_rank),
                    "end_rank": int(package_end_rank),
                },
                "technical_rows": int(len(technical_df)),
                "final_rows": int(len(final_df)),
                "selected_chart_tickers": selected_chart_rows.to_dict(orient="records"),
                "selected_package_tickers": package_df[["Ticker", "screen_rank"]].to_dict(orient="records"),
                "final_ranked_stocks": final_df.to_dict(orient="records"),
                "artifact_paths": {
                    "final_feather": str(feather_path),
                    "final_csv": str(csv_path),
                    "screening_pdf": str(pdf_path),
                    "chart_manifest_csv": str(chart_manifest_path),
                    "package_manifest_csv": str(package_manifest_path),
                },
            },
        )

    print("Done.")
    print(f"Technical rows: {len(technical_df)}")
    print(f"Insider rows with score > {cfg.insider_score_threshold}: {(insider_ranked_df['insider_score_60d'] > cfg.insider_score_threshold).sum()}")
    print(f"Saved insider ranking: {insider_ranked_path}")
    print(f"Saved combined ranking: {combined_ranked_path}")
    print(f"Final deduped rows: {len(final_df)}")
    print(f"Saved final feather: {feather_path}")
    print(f"Saved final csv: {csv_path}")
    print(f"Saved chart manifest: {chart_manifest_path}")
    print(f"Saved charts pdf: {pdf_path}")
    print(f"Saved data package manifest: {package_manifest_path}")
    print(f"Saved data packages root: {AGENTS_DATA_PACKAGE_DIR}")


if __name__ == "__main__":
    main()
