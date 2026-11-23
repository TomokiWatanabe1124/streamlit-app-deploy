"""東証の制限値幅（ストップ高・ストップ安）の計算。

基準値段（通常は前営業日の終値）に応じて、1日に動ける値幅が決まっている。
出典: 日本取引所グループ「制限値幅」の表（通常時の値幅）。
制限値幅の拡大措置、新規上場日、権利落ちなどの例外は考慮していない。
"""

from __future__ import annotations

import math

# (基準値段の上限（未満）, 制限値幅)
_LIMIT_TABLE: list[tuple[float, int]] = [
    (100, 30),
    (200, 50),
    (500, 80),
    (700, 100),
    (1_000, 150),
    (1_500, 300),
    (2_000, 400),
    (3_000, 500),
    (5_000, 700),
    (7_000, 1_000),
    (10_000, 1_500),
    (15_000, 3_000),
    (20_000, 4_000),
    (30_000, 5_000),
    (50_000, 7_000),
    (70_000, 10_000),
    (100_000, 15_000),
    (150_000, 30_000),
    (200_000, 40_000),
    (300_000, 50_000),
    (500_000, 70_000),
    (700_000, 100_000),
    (1_000_000, 150_000),
    (1_500_000, 300_000),
    (2_000_000, 400_000),
    (3_000_000, 500_000),
    (5_000_000, 700_000),
    (7_000_000, 1_000_000),
    (10_000_000, 1_500_000),
    (15_000_000, 3_000_000),
    (20_000_000, 4_000_000),
    (30_000_000, 5_000_000),
    (50_000_000, 7_000_000),
    (math.inf, 10_000_000),
]


def limit_width(base_price: float) -> int:
    """基準値段に対する制限値幅を返す。"""
    for upper, width in _LIMIT_TABLE:
        if base_price < upper:
            return width
    raise ValueError(base_price)  # pragma: no cover - math.inf で必ず止まる


def limit_prices(base_price: float) -> tuple[float, float]:
    """(ストップ高の値段, ストップ安の値段) を返す。ストップ安は 1 円未満にならない。"""
    w = limit_width(base_price)
    return base_price + w, max(base_price - w, 1.0)


def classify_limit(prev_close: float, high: float, low: float, close: float) -> str:
    """当日の値動きがストップ高・安に達したかを判定する。

    戻り値: "S高引け" / "S高到達" / "S安引け" / "S安到達" / ""
    """
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (prev_close, high, low, close)):
        return ""
    if prev_close <= 0:
        return ""
    up, down = limit_prices(prev_close)
    eps = 1e-6
    if close >= up - eps:
        return "S高引け"
    if close <= down + eps:
        return "S安引け"
    if high >= up - eps:
        return "S高到達"
    if low <= down + eps:
        return "S安到達"
    return ""
