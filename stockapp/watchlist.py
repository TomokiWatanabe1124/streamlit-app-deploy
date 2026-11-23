"""ウォッチ銘柄リスト（watchlist.csv）の読み込み。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "watchlist.csv"


def normalize_code(code: object) -> str:
    """証券コードを 4 桁（英字入りコード含む）の文字列に揃える。

    TDnet は 5 桁（末尾 0）で表すので、その形式も受け付ける。
    """
    s = str(code).strip().upper()
    if s.endswith(".T"):
        s = s[:-2]
    if s.endswith(".0"):  # Excel などで数値として読まれた場合
        s = s[:-2]
    if len(s) == 5 and s.endswith("0"):
        s = s[:4]
    return s


def to_yahoo(code: str) -> str:
    return f"{normalize_code(code)}.T"


def load_watchlist(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """code, name, memo の DataFrame を返す。空行・重複は除く。"""
    df = pd.read_csv(path, dtype=str, comment="#").fillna("")
    for col in ("code", "name", "memo"):
        if col not in df.columns:
            df[col] = ""
    df["code"] = df["code"].map(normalize_code)
    df = df[df["code"] != ""].drop_duplicates("code")
    return df[["code", "name", "memo"]].reset_index(drop=True)
