import pandas as pd
import pytest

from stockapp.price_limits import classify_limit, limit_prices, limit_width
from stockapp.tdnet import categorize, parse_page
from stockapp.watchlist import normalize_code


@pytest.mark.parametrize(
    "base,width",
    [(77, 30), (99, 30), (100, 50), (499, 80), (500, 100), (999, 150), (1000, 300),
     (1783, 400), (2999, 500), (3000, 700), (9999, 1500), (49_999, 7_000), (60_000_000, 10_000_000)],
)
def test_limit_width(base, width):
    assert limit_width(base) == width


def test_limit_prices_floor():
    assert limit_prices(10) == (40, 1.0)


@pytest.mark.parametrize(
    "prev,high,low,close,expected",
    [
        (77, 107, 76, 107, "S高引け"),   # 4586 の実例
        (139, 185, 90, 185, ""),          # 上限 189 に届かず
        (1000, 1300, 990, 1250, "S高到達"),
        (1000, 1010, 700, 700, "S安引け"),
        (1000, 1010, 700, 750, "S安到達"),
        (1000, 1010, 990, 1000, ""),
        (float("nan"), 1, 1, 1, ""),
    ],
)
def test_classify_limit(prev, high, low, close, expected):
    assert classify_limit(prev, high, low, close) == expected


@pytest.mark.parametrize("raw,code", [("2801", "2801"), ("28010", "2801"), ("2801.T", "2801"),
                                       (2801, "2801"), ("130a", "130A"), ("130A0", "130A"), ("2801.0", "2801")])
def test_normalize_code(raw, code):
    assert normalize_code(raw) == code


@pytest.mark.parametrize(
    "title,cat",
    [
        ("2026年12月期通期業績予想の修正に関するお知らせ", "業績修正"),
        ("通期業績計画の修正に関するお知らせ", "業績修正"),
        ("業績予想及び配当予想の修正に関するお知らせ", "業績修正"),
        ("配当予想の修正（増配）に関するお知らせ", "配当"),
        ("2027年3月期 第2四半期（中間期）決算短信〔日本基準〕（連結）", "決算"),
        ("自己株式の取得状況に関するお知らせ", ""),
    ],
)
def test_categorize(title, cat):
    assert categorize(title) == cat


SAMPLE = """
<html><body>
<div>1～2件&nbsp;/&nbsp;全2件</div>
<table id="main-list-table">
<tr><td class="oddnew-L kjTime">15:30</td><td class="oddnew-M kjCode">28010</td>
<td class="oddnew-M kjName">キッコーマン</td>
<td class="oddnew-M kjTitle"><a href="140120261120000001.pdf">業績予想の修正に関するお知らせ</a></td></tr>
<tr><td class="evennew-L kjTime">16:00</td><td class="evennew-M kjCode">84100</td>
<td class="evennew-M kjName">セブン銀行</td>
<td class="evennew-M kjTitle"><a href="140120261120000002.pdf">自己株式の取得状況に関するお知らせ</a></td></tr>
</table></body></html>
"""


def test_parse_page():
    rows, has_next = parse_page(SAMPLE)
    assert not has_next
    df = pd.DataFrame(rows)
    assert df["code"].tolist() == ["2801", "8410"]
    assert df.loc[0, "category"] == "業績修正"
    assert df.loc[0, "url"].endswith("140120261120000001.pdf")


def test_parse_page_has_next():
    _, has_next = parse_page(SAMPLE.replace("1～2件&nbsp;/&nbsp;全2件", "1～100件&nbsp;/&nbsp;全284件"))
    assert has_next


# --- market -----------------------------------------------------------------

from stockapp.market import enrich, parse_chart, parse_spark  # noqa: E402

SPARK = {"spark": {"result": [
    {"symbol": "4586.T", "response": [{"meta": {
        "symbol": "4586.T", "regularMarketPrice": 107.0, "fulldayChange": 30.0,
        "regularMarketChangePercent": 38.961, "regularMarketDayHigh": 107.0,
        "regularMarketDayLow": 76.0, "regularMarketVolume": 48351100, "regularMarketTime": 1795156200}}]},
    {"symbol": "9999.T", "response": []},
], "error": None}}


def test_parse_spark_and_enrich():
    rows = parse_spark(SPARK)
    assert len(rows) == 1
    assert rows[0]["prev_close"] == 77.0
    df = enrich(pd.DataFrame(rows))
    assert df.loc[0, "code"] == "4586"
    assert df.loc[0, "limit"] == "S高引け"
    assert pd.isna(df.loc[0, "volume_ratio"])  # 平均出来高なしでも落ちない


def test_parse_chart_drops_empty_days():
    payload = {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "Asia/Tokyo"},
        "timestamp": [1795046400, 1795132800, 1795392000],
        "indicators": {"quote": [{"open": [1, 2, None], "high": [1, 2, None], "low": [1, 2, None],
                                  "close": [1.0, 2.0, None], "volume": [10, 20, None]}]},
    }]}}
    df = parse_chart(payload)
    assert len(df) == 2
    assert str(df.index[-1].date()) == "2026-11-20"


# --- company ----------------------------------------------------------------

import numpy as np  # noqa: E402

from stockapp import company  # noqa: E402


def _hist(prices, vol=None):
    idx = pd.bdate_range("2025-11-01", periods=len(prices))
    p = pd.Series(prices, index=idx, dtype="float64")
    return pd.DataFrame({"Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p,
                         "Volume": vol if vol is not None else np.full(len(p), 1000.0)})


def test_analyze_uptrend_high_zone():
    h = _hist(np.linspace(1000, 2000, 260))
    v = company.analyze(h)
    assert v.trend == "上昇トレンド"
    assert v.range_pos > 0.9
    assert v.rsi14 == 100.0
    texts = " ".join(t for _, t in v.comments)
    assert "上昇トレンド" in texts and "高値圏" in texts and "買われすぎ" in texts


def test_analyze_downtrend_low_zone_and_relative():
    h = _hist(np.linspace(2000, 1000, 260))
    bench = _hist(np.full(260, 100.0))
    v = company.analyze(h, bench)
    assert v.trend == "下降トレンド"
    assert v.range_pos < 0.1
    assert v.rel_3m < -5
    assert any("TOPIX を" in t and "下回って" in t for _, t in v.comments)


def test_analyze_volume_spike_and_short_history():
    vol = np.r_[np.full(255, 1000.0), np.full(5, 3000.0)]
    v = company.analyze(_hist(np.full(260, 500.0), vol))
    assert v.vol_ratio_5_60 > 1.5
    assert any("出来高" in t for _, t in v.comments)
    assert company.analyze(_hist(np.full(10, 500.0))) is None


def test_build_prompt_and_links():
    v = company.analyze(_hist(np.linspace(1000, 1200, 260)))
    disc = pd.DataFrame([{"day": "2026-11-20", "title": "業績予想の修正に関するお知らせ"}])
    p = company.build_prompt("2801", "キッコーマン", "プライム", "食料品", v,
                             {"per": 24.3, "dividend_yield": 1.4}, disc, "2027-02-04")
    assert "キッコーマン" in p and "PER（実績）: 24.3倍" in p and "配当利回り: 1.40%" in p
    assert "業績予想の修正" in p and "強み" in p and "リスク" in p
    links = dict((n, u) for n, u, _ in company.external_links("2801", "キッコーマン"))
    assert links["株探"].endswith("code=2801")
