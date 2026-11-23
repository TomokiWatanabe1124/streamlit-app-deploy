"""Yahoo Finance からの株価取得。

- 全銘柄の当日の相場: quote API を 200 銘柄ずつまとめて呼ぶ（全銘柄で数秒）
- チャート用の日足: yfinance.download
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf
from yfinance.data import YfData

from .price_limits import classify_limit
from .watchlist import normalize_code

QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
BATCH = 200

# 相場サマリーに出す指標（表示名, Yahoo のシンボル, 補足）
INDICES = [
    ("日経平均", "^N225", ""),
    ("TOPIX（ETF 1306）", "1306.T", "TOPIX 指数は取得できないため連動 ETF で代用"),
    ("グロース250（ETF 2516）", "2516.T", "東証グロース市場 250 指数連動 ETF"),
    ("ドル円", "JPY=X", ""),
]

_QUOTE_FIELDS = {
    "symbol": "symbol",
    "regularMarketPrice": "close",
    "regularMarketPreviousClose": "prev_close",
    "regularMarketChange": "change",
    "regularMarketChangePercent": "change_pct",
    "regularMarketVolume": "volume",
    "regularMarketDayHigh": "high",
    "regularMarketDayLow": "low",
    "regularMarketOpen": "open",
    "averageDailyVolume3Month": "avg_volume_3m",
    "marketCap": "market_cap",
    "regularMarketTime": "time",
}


def fetch_quotes(symbols: list[str], retries: int = 3) -> pd.DataFrame:
    """シンボルのリストの最新相場を返す。"""
    yd = YfData()
    rows: list[dict] = []
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i : i + BATCH]
        for attempt in range(retries):
            try:
                r = yd.get(QUOTE_URL, params={"symbols": ",".join(chunk)})
                r.raise_for_status()
                rows += r.json()["quoteResponse"]["result"]
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=list(_QUOTE_FIELDS.values()))
    for src in _QUOTE_FIELDS:
        if src not in df.columns:
            df[src] = pd.NA
    df = df[list(_QUOTE_FIELDS)].rename(columns=_QUOTE_FIELDS)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Tokyo")
    df["date"] = df["time"].dt.date
    return df


def enrich(quotes: pd.DataFrame) -> pd.DataFrame:
    """コード、出来高倍率、ストップ高・安の判定を付ける。"""
    df = quotes.copy()
    df["code"] = df["symbol"].map(normalize_code)
    df["volume_ratio"] = df["volume"] / df["avg_volume_3m"].where(df["avg_volume_3m"] > 0)
    df["limit"] = [
        classify_limit(p, h, lo, c)
        for p, h, lo, c in zip(df["prev_close"], df["high"], df["low"], df["close"])
    ]
    return df


def fetch_market(codes: list[str]) -> pd.DataFrame:
    """全銘柄の相場を取得し、最新営業日に売買のあった銘柄だけを返す。"""
    df = enrich(fetch_quotes([f"{c}.T" for c in codes]))
    if df.empty:
        return df
    latest = df["date"].max()
    return df[df["date"] == latest].reset_index(drop=True)


def fetch_history(symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """日足（OHLCV）をシンボルごとに返す。データのない日（休場日）は除く。"""
    raw = yf.download(
        symbols, period=period, interval="1d", group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            d = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        d = d.dropna(subset=["Close"])
        if not d.empty:
            out[s] = d
    return out
