"""日本取引所グループ（JPX）が公開しているファイルの取得。

- 東証上場銘柄一覧（毎月更新）
- 決算発表予定日（決算期末の月ごとの Excel。おおむね毎週更新）
"""

from __future__ import annotations

import io
import re
from urllib.parse import urljoin

import pandas as pd
import requests

from .watchlist import normalize_code

BASE = "https://www.jpx.co.jp"
LISTING_PAGE = BASE + "/markets/statistics-equities/misc/01.html"
EARNINGS_PAGE = BASE + "/listing/event-schedules/financial-announcement/index.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal stock checker)"}
TIMEOUT = 30


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def _excel_links(page_url: str, pattern: str) -> list[str]:
    html = _get(page_url).text
    hrefs = re.findall(r'href="([^"]+\.xlsx?)"', html)
    return [urljoin(page_url, h) for h in hrefs if re.search(pattern, h)]


# ---------------------------------------------------------------------------
# 上場銘柄一覧
# ---------------------------------------------------------------------------

STOCK_SEGMENTS = ("プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）")


def parse_listing(content: bytes) -> pd.DataFrame:
    """data_j.xlsx を code, name, market, sector の DataFrame にする（内国株式のみ）。"""
    df = pd.read_excel(io.BytesIO(content), dtype={"コード": str})
    df = df[df["市場・商品区分"].isin(STOCK_SEGMENTS)].copy()
    df["code"] = df["コード"].map(normalize_code)
    df["market"] = df["市場・商品区分"].str.replace("（内国株式）", "", regex=False)
    df = df.rename(columns={"銘柄名": "name", "33業種区分": "sector"})
    return df[["code", "name", "market", "sector"]].reset_index(drop=True)


def fetch_listing() -> pd.DataFrame:
    links = _excel_links(LISTING_PAGE, r"data_j\.xlsx?$")
    if not links:
        raise RuntimeError("JPX の上場銘柄一覧ファイルが見つかりませんでした")
    return parse_listing(_get(links[0]).content)


# ---------------------------------------------------------------------------
# 決算発表予定日
# ---------------------------------------------------------------------------

def parse_earnings(content: bytes) -> pd.DataFrame:
    """kessanMM_MMDD.xlsx を date, code, name, period, kind, sector, market にする。"""
    raw = pd.read_excel(io.BytesIO(content), header=None)
    # 見出し行（「決算発表予定日」で始まる行）を探す
    header_idx = None
    for i, v in raw[0].items():
        if isinstance(v, str) and v.startswith("決算発表予定日"):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame(columns=["date", "code", "name", "period", "kind", "sector", "market"])
    body = raw.iloc[header_idx + 1 :, :10]
    body.columns = ["date", "code", "name", "name_en", "period", "sector", "sector_en", "kind", "kind_en", "market"]
    body = body.assign(date=pd.to_datetime(body["date"], errors="coerce"))
    body = body.dropna(subset=["date", "code"])
    body["code"] = body["code"].map(normalize_code)
    body["date"] = body["date"].dt.date
    body["period"] = pd.to_datetime(body["period"], errors="coerce").dt.strftime("%Y-%m")
    return body[["date", "code", "name", "period", "kind", "sector", "market"]].reset_index(drop=True)


def fetch_earnings_schedule() -> pd.DataFrame:
    """公開中の決算発表予定ファイルをすべて読み込んで結合する。"""
    links = _excel_links(EARNINGS_PAGE, r"kessan.*\.xlsx?$")
    if not links:
        raise RuntimeError("JPX の決算発表予定日ファイルが見つかりませんでした")
    frames = [parse_earnings(_get(u).content) for u in links]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)
