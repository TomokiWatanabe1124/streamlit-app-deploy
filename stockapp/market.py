"""Yahoo Finance からの株価取得。

Streamlit Community Cloud などの共用サーバーでは、Yahoo の認証トークン（crumb）の取得が
失敗しやすく、quote API は 401 を返すことがある。そのため、トークンが不要な API を主に使う。

- 全銘柄の当日の相場: spark API（20 銘柄ずつ、並列で呼ぶ。全銘柄で数秒）
- 3 か月平均出来高: quote API（トークンが必要。取れなかった場合は省略する）
- チャート用の日足: chart API
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .price_limits import classify_limit
from .watchlist import normalize_code

SPARK_URL = "https://query1.finance.yahoo.com/v7/finance/spark"
QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
TIMEOUT = 30
SPARK_BATCH = 20  # spark API の上限
WORKERS = 6

# 相場サマリーに出す指標（表示名, Yahoo のシンボル, 補足）
INDICES = [
    ("日経平均", "^N225", ""),
    ("TOPIX（ETF 1306）", "1306.T", "TOPIX 指数は取得できないため連動 ETF で代用"),
    ("グロース250（ETF 2516）", "2516.T", "東証グロース市場 250 指数連動 ETF"),
    ("ドル円", "JPY=X", ""),
]

QUOTE_COLUMNS = ["symbol", "close", "prev_close", "change", "change_pct", "volume",
                 "high", "low", "avg_volume_3m", "time", "date"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get_json(session: requests.Session, url: str, params: dict, retries: int = 4) -> dict:
    """429（アクセス過多）や一時的なエラーは、待ってから再試行する。"""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise  # 401/404 などは再試行しても変わらない
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# 当日の相場（全銘柄）
# ---------------------------------------------------------------------------

def parse_spark(payload: dict) -> list[dict]:
    """spark API の応答から、銘柄ごとの当日の相場を取り出す。"""
    rows = []
    for item in (payload.get("spark") or {}).get("result") or []:
        resp = (item.get("response") or [None])[0]
        if not resp:
            continue
        m = resp.get("meta") or {}
        price = m.get("regularMarketPrice")
        change = m.get("fulldayChange")
        if price is None:
            continue
        prev = price - change if change is not None else None
        pct = m.get("regularMarketChangePercent", m.get("fulldayChangePercent"))
        rows.append(
            {
                "symbol": m.get("symbol") or item.get("symbol"),
                "close": price,
                "prev_close": prev,
                "change": change,
                "change_pct": pct,
                "volume": m.get("regularMarketVolume"),
                "high": m.get("regularMarketDayHigh"),
                "low": m.get("regularMarketDayLow"),
                "time": m.get("regularMarketTime"),
            }
        )
    return rows


def fetch_spark(symbols: list[str]) -> pd.DataFrame:
    session = _session()
    chunks = [symbols[i : i + SPARK_BATCH] for i in range(0, len(symbols), SPARK_BATCH)]

    def one(chunk: list[str]) -> list[dict]:
        payload = _get_json(session, SPARK_URL, {"symbols": ",".join(chunk), "range": "1d", "interval": "1d"})
        return parse_spark(payload)

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(one, chunks))
    rows = [r for chunk in results for r in chunk]
    df = pd.DataFrame(rows, columns=[c for c in QUOTE_COLUMNS if c not in ("avg_volume_3m", "date")])
    for c in ("close", "prev_close", "change", "change_pct", "volume", "high", "low"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Tokyo")
    df["date"] = df["time"].dt.date
    return df


def fetch_avg_volume(symbols: list[str]) -> pd.Series:
    """3 か月平均出来高（quote API）。認証トークンが取れない環境では空を返す。"""
    try:
        from yfinance.data import YfData

        yd = YfData()
        out: dict[str, float] = {}
        for i in range(0, len(symbols), 200):
            r = yd.get(QUOTE_URL, params={"symbols": ",".join(symbols[i : i + 200])})
            r.raise_for_status()
            for q in r.json()["quoteResponse"]["result"]:
                if q.get("averageDailyVolume3Month"):
                    out[q["symbol"]] = q["averageDailyVolume3Month"]
        return pd.Series(out, dtype="float64")
    except Exception:
        return pd.Series(dtype="float64")


def enrich(quotes: pd.DataFrame) -> pd.DataFrame:
    """コード、出来高倍率、ストップ高・安の判定を付ける。"""
    df = quotes.copy()
    df["code"] = df["symbol"].map(normalize_code)
    if "avg_volume_3m" not in df:
        df["avg_volume_3m"] = float("nan")
    df["volume_ratio"] = df["volume"] / df["avg_volume_3m"].where(df["avg_volume_3m"] > 0)
    df["limit"] = [
        classify_limit(p, h, lo, c)
        for p, h, lo, c in zip(df["prev_close"], df["high"], df["low"], df["close"])
    ]
    return df


def fetch_market(codes: list[str]) -> pd.DataFrame:
    """全銘柄の相場を取得し、最新営業日に売買のあった銘柄だけを返す。"""
    symbols = [f"{c}.T" for c in codes]
    df = fetch_spark(symbols)
    if df.empty:
        return enrich(df.assign(avg_volume_3m=pd.Series(dtype="float64")))
    df["avg_volume_3m"] = df["symbol"].map(fetch_avg_volume(symbols))
    df = enrich(df)
    latest = df["date"].max()
    return df[df["date"] == latest].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 日足（チャート・指数）
# ---------------------------------------------------------------------------

def parse_chart(payload: dict) -> pd.DataFrame:
    res = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    q = res["indicators"]["quote"][0]
    tz = (res.get("meta") or {}).get("exchangeTimezoneName", "Asia/Tokyo")
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    df = pd.DataFrame(
        {"Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
         "Close": q.get("close"), "Volume": q.get("volume")},
        index=idx,
    )
    df = df.dropna(subset=["Close"])
    return df[~df.index.duplicated(keep="last")]


def fetch_history(symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """日足（OHLCV）をシンボルごとに返す。データのない日（休場日）は除く。"""
    session = _session()

    def one(sym: str) -> tuple[str, pd.DataFrame | None]:
        try:
            payload = _get_json(session, CHART_URL.format(symbol=sym), {"range": period, "interval": "1d"})
            return sym, parse_chart(payload)
        except Exception:
            return sym, None

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(one, symbols))
    return {s: d for s, d in results if d is not None and not d.empty}
