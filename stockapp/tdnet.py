"""TDnet（適時開示情報閲覧サービス）の開示一覧の取得。

1 日分の一覧は 100 件ずつのページに分かれている。直近 1 か月分ほどが公開されている。
"""

from __future__ import annotations

import datetime as dt
import functools
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .watchlist import normalize_code

BASE = "https://www.release.tdnet.info/inbs/"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal stock checker)"}
TIMEOUT = 30
MAX_PAGES = 30  # 決算集中日でも 1 日 3,000 件程度

# 業績修正とみなす表題のパターン
EARNINGS_REVISION = re.compile(r"業績予想.{0,10}修正|業績.{0,6}修正|上方修正|下方修正|予想値と.{0,10}差異")
DIVIDEND_REVISION = re.compile(r"配当予想.{0,10}修正|増配|減配|復配|無配")
EARNINGS_REPORT = re.compile(r"決算短信")

COLUMNS = ["time", "code", "name", "title", "url", "category"]


def categorize(title: str) -> str:
    if EARNINGS_REVISION.search(title):
        return "業績修正"
    if DIVIDEND_REVISION.search(title):
        return "配当"
    if EARNINGS_REPORT.search(title):
        return "決算"
    return ""


def parse_page(html: bytes | str) -> tuple[list[dict], bool]:
    """1 ページを解析し、(行のリスト, 次のページがあるか) を返す。"""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="main-list-table")
    rows: list[dict] = []
    if table is not None:
        for tr in table.find_all("tr"):
            cells = {c: tr.find("td", class_=re.compile(c)) for c in ("kjTime", "kjCode", "kjName", "kjTitle")}
            if not all(cells.values()):
                continue
            a = cells["kjTitle"].find("a")
            title = cells["kjTitle"].get_text(strip=True)
            rows.append(
                {
                    "time": cells["kjTime"].get_text(strip=True),
                    "code": normalize_code(cells["kjCode"].get_text(strip=True)),
                    "name": cells["kjName"].get_text(strip=True),
                    "title": title,
                    "url": BASE + a["href"] if a and a.get("href") else "",
                    "category": categorize(title),
                }
            )
    # 「次へ」ボタン or 現在ページより大きいページ番号があれば続きがある
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(\d+)～(\d+)件\s*/\s*全(\d+)件", text)
    has_next = bool(m) and int(m.group(2)) < int(m.group(3))
    return rows, has_next


@functools.lru_cache(maxsize=64)
def _fetch_past_day(day: dt.date) -> pd.DataFrame:
    """過去の日の一覧は後から変わらないので、プロセス内で使い回す。"""
    return fetch_day(day)


def fetch_days(days: list[dt.date], today: dt.date, workers: int = 4) -> pd.DataFrame:
    """複数日の一覧をまとめて取得する（day 列付き、新しい順）。"""

    def one(d: dt.date) -> pd.DataFrame:
        df = _fetch_past_day(d) if d < today else fetch_day(d)
        return df.assign(day=d)

    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(one, sorted(days, reverse=True)))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS + ["day"])
    return pd.concat(frames, ignore_index=True)


def fetch_day(day: dt.date) -> pd.DataFrame:
    """指定日の適時開示一覧を返す（新しい順）。公開されていない日は空。"""
    rows: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{BASE}I_list_{page:03d}_{day:%Y%m%d}.html"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404:
            break
        r.raise_for_status()
        page_rows, has_next = parse_page(r.content)
        rows += page_rows
        if not has_next:
            break
        time.sleep(0.3)  # サーバーに負荷をかけない
    return pd.DataFrame(rows, columns=COLUMNS)
