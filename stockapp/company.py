"""企業情報タブ用: 外部リンク、株価水準の指標とコメント、Claude.ai 用の質問文。

株価水準のコメントは、テクニカル指標に決まったルールを当てはめて作る機械的な目安。
AI は使わないので無料で動く。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote_plus

import pandas as pd

# ---------------------------------------------------------------------------
# 外部リンク
# ---------------------------------------------------------------------------


def external_links(code: str, name: str) -> list[tuple[str, str, str]]:
    """(サイト名, URL, 何が見られるか) のリスト。"""
    q = quote_plus(f"{name} IR 投資家情報")
    return [
        ("Yahoo!ファイナンス", f"https://finance.yahoo.co.jp/quote/{code}.T", "株価・PER/PBR・配当・掲示板"),
        ("株探", f"https://kabutan.jp/stock/?code={code}", "決算の要点・ニュース・業績推移"),
        ("IR BANK", f"https://irbank.net/{code}", "過去10年以上の業績・財務・配当の推移"),
        ("日本経済新聞", f"https://www.nikkei.com/nkd/company/?scode={code}", "企業ニュース・会社概要"),
        ("会社の IR ページを検索", f"https://www.google.com/search?q={q}", "決算説明資料・中期経営計画・統合報告書"),
        ("EDINET（有価証券報告書）", "https://disclosure2.edinet-fsa.go.jp/", "有価証券報告書の原本（会社名で検索）"),
    ]


# ---------------------------------------------------------------------------
# 株価水準の指標
# ---------------------------------------------------------------------------


@dataclass
class PriceView:
    close: float
    date: pd.Timestamp
    high_52w: float | None = None
    low_52w: float | None = None
    range_pos: float | None = None  # 52週レンジ内の位置 0〜1
    dev_ma25: float | None = None  # 25日線からの乖離率 %
    dev_ma75: float | None = None
    rsi14: float | None = None
    ret_1m: float | None = None  # 騰落率 %
    ret_3m: float | None = None
    ret_1y: float | None = None
    rel_3m: float | None = None  # 3か月騰落率の TOPIX との差（ポイント）
    vol_ratio_5_60: float | None = None  # 直近5日平均出来高 ÷ 60日平均
    trend: str = ""
    comments: list[tuple[str, str]] = field(default_factory=list)  # (アイコン, 文)


def _ret(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return (close.iloc[-1] / close.iloc[-1 - days] - 1) * 100


def rsi(close: pd.Series, n: int = 14) -> float | None:
    """RSI（ワイルダーの平滑化）。"""
    if len(close) <= n:
        return None
    diff = close.diff().dropna()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return 100 - 100 / (1 + gain / loss)


def analyze(hist: pd.DataFrame, bench: pd.DataFrame | None = None) -> PriceView | None:
    """1年分の日足から株価水準の指標とコメントを作る。"""
    if hist is None or len(hist) < 30:
        return None
    c = hist["Close"]
    v = PriceView(close=float(c.iloc[-1]), date=c.index[-1])

    last_year = hist.tail(250)
    v.high_52w = float(last_year["High"].max())
    v.low_52w = float(last_year["Low"].min())
    if v.high_52w > v.low_52w:
        v.range_pos = (v.close - v.low_52w) / (v.high_52w - v.low_52w)

    ma25 = c.rolling(25).mean().iloc[-1]
    ma75 = c.rolling(75).mean().iloc[-1] if len(c) >= 75 else None
    if pd.notna(ma25):
        v.dev_ma25 = (v.close / ma25 - 1) * 100
    if ma75 is not None and pd.notna(ma75):
        v.dev_ma75 = (v.close / ma75 - 1) * 100
    v.rsi14 = rsi(c)
    v.ret_1m, v.ret_3m = _ret(c, 20), _ret(c, 60)
    # 1年分の日足は営業日の数が年によって違う（240〜250本程度）ので、取得した期間の先頭と比べる
    if len(c) >= 200:
        v.ret_1y = (c.iloc[-1] / c.iloc[max(0, len(c) - 246)] - 1) * 100

    if bench is not None and len(bench) > 60 and v.ret_3m is not None:
        b = _ret(bench["Close"], 60)
        if b is not None:
            v.rel_3m = v.ret_3m - b

    vol = hist["Volume"].dropna()
    if len(vol) >= 65 and vol.tail(60).mean() > 0:
        v.vol_ratio_5_60 = vol.tail(5).mean() / vol.tail(60).mean()

    # トレンド
    if ma75 is not None and pd.notna(ma25) and pd.notna(ma75):
        if v.close > ma25 > ma75:
            v.trend = "上昇トレンド"
        elif v.close < ma25 < ma75:
            v.trend = "下降トレンド"
        else:
            v.trend = "もみ合い・方向感を探る局面"

    v.comments = make_comments(v)
    return v


def make_comments(v: PriceView) -> list[tuple[str, str]]:
    """指標にルールを当てはめて、株価水準の見方を文章にする。"""
    out: list[tuple[str, str]] = []

    if v.trend == "上昇トレンド":
        out.append(("📈", "株価が25日線・75日線の上にあり、上昇トレンドが続いています。"))
    elif v.trend == "下降トレンド":
        out.append(("📉", "株価が25日線・75日線の下にあり、下降トレンドが続いています。"))
    elif v.trend:
        out.append(("↔️", "移動平均線の並びがそろっておらず、方向感を探る局面です。"))

    if v.range_pos is not None:
        pct = v.range_pos * 100
        if v.range_pos >= 0.9:
            out.append(("🔝", f"52週の高値圏にあります（年間の値幅のうち上から{100 - pct:.0f}%の位置）。高値更新の勢いと、利益確定売りの両方に注意する水準です。"))
        elif v.range_pos <= 0.1:
            out.append(("🔻", f"52週の安値圏にあります（年間の値幅のうち下から{pct:.0f}%の位置）。割安感が出ている一方、下げ止まりの確認が必要な水準です。"))
        else:
            out.append(("📏", f"52週の値幅の中では、下から{pct:.0f}%の位置にあります。"))

    if v.dev_ma25 is not None:
        d = v.dev_ma25
        if d >= 10:
            out.append(("🔥", f"25日線を{d:+.1f}%上回っており、短期的には過熱感が強い水準です。"))
        elif d >= 5:
            out.append(("♨️", f"25日線を{d:+.1f}%上回っており、短期的にはやや過熱気味です。"))
        elif d <= -10:
            out.append(("🧊", f"25日線を{d:.1f}%下回っており、短期的には売られすぎ感が強い水準です。"))
        elif d <= -5:
            out.append(("❄️", f"25日線を{d:.1f}%下回っており、短期的にはやや売られすぎです。"))
        else:
            out.append(("⚖️", f"25日線との乖離は{d:+.1f}%で、短期的な過熱感はありません。"))

    if v.rsi14 is not None:
        if v.rsi14 >= 70:
            out.append(("⚠️", f"RSI(14日)は{v.rsi14:.0f}で、買われすぎの目安（70以上）に入っています。"))
        elif v.rsi14 <= 30:
            out.append(("⚠️", f"RSI(14日)は{v.rsi14:.0f}で、売られすぎの目安（30以下）に入っています。"))

    if v.rel_3m is not None:
        if v.rel_3m >= 5:
            out.append(("💪", f"直近3か月は TOPIX を{v.rel_3m:.1f}ポイント上回っており、市場全体より強い動きです。"))
        elif v.rel_3m <= -5:
            out.append(("🐢", f"直近3か月は TOPIX を{-v.rel_3m:.1f}ポイント下回っており、市場全体より弱い動きです。"))
        else:
            out.append(("🤝", "直近3か月は、おおむね TOPIX と同じような動きです。"))

    if v.vol_ratio_5_60 is not None:
        if v.vol_ratio_5_60 >= 1.5:
            out.append(("📢", f"直近5日の出来高が平常時の{v.vol_ratio_5_60:.1f}倍に増えており、注目が集まっています。"))
        elif v.vol_ratio_5_60 <= 0.6:
            out.append(("💤", "直近5日の出来高は平常時より少なく、関心が薄れています。"))

    return out


# ---------------------------------------------------------------------------
# Claude.ai に貼る質問文
# ---------------------------------------------------------------------------


def _fmt(v: float | None, fmt: str, suffix: str = "") -> str:
    return "不明" if v is None or pd.isna(v) else f"{v:{fmt}}{suffix}"


def build_prompt(
    code: str,
    name: str,
    market: str,
    sector: str,
    view: PriceView | None,
    valuation: dict,
    disclosures: pd.DataFrame,
    next_earnings: str,
) -> str:
    lines = [
        f"東証に上場している「{name}」（証券コード {code}、{market}市場、{sector}）について、個人投資家向けに分析してください。",
        "",
        "## アプリで集めた情報",
    ]
    if view is not None:
        lines += [
            f"- 株価: {view.close:,.1f}円（{view.date:%Y-%m-%d} 終値）",
            f"- 52週高値 / 安値: {_fmt(view.high_52w, ',.1f', '円')} / {_fmt(view.low_52w, ',.1f', '円')}",
            f"- 騰落率: 1か月 {_fmt(view.ret_1m, '+.1f', '%')}、3か月 {_fmt(view.ret_3m, '+.1f', '%')}、1年 {_fmt(view.ret_1y, '+.1f', '%')}",
            f"- 3か月騰落率の TOPIX との差: {_fmt(view.rel_3m, '+.1f', 'ポイント')}",
            f"- 25日線乖離率 {_fmt(view.dev_ma25, '+.1f', '%')}、75日線乖離率 {_fmt(view.dev_ma75, '+.1f', '%')}、RSI(14) {_fmt(view.rsi14, '.0f')}",
            f"- トレンド: {view.trend or '不明'}",
        ]
    labels = {"per": ("PER（実績）", ".1f", "倍"), "forward_per": ("PER（予想）", ".1f", "倍"),
              "pbr": ("PBR", ".2f", "倍"), "dividend_yield": ("配当利回り", ".2f", "%"),
              "market_cap_oku": ("時価総額", ",.0f", "億円")}
    for k, (lab, f, suf) in labels.items():
        if valuation.get(k) is not None:
            lines.append(f"- {lab}: {valuation[k]:{f}}{suf}")
    lines.append(f"- 次回決算発表予定: {next_earnings}")
    if disclosures is not None and not disclosures.empty:
        lines.append("- 最近の適時開示:")
        for _, r in disclosures.head(15).iterrows():
            lines.append(f"  - {r['day']} {r['title']}")
    else:
        lines.append("- 最近の適時開示: なし（アプリで確認した期間内）")

    lines += [
        "",
        "## 知りたいこと",
        "1. 事業の概要（何で稼いでいる会社か、3〜4行で）",
        "2. 強み（競争優位、収益力、財務、成長ドライバーなど）を3〜5点",
        "3. リスク・懸念点（業績、財務、競争環境、為替や原材料など）を3〜5点",
        "4. 今の株価水準の見方（上の株価指標と、PER・PBR・配当利回りを同業他社や過去の水準と比べて、割高・割安・妥当のどれに近いか）",
        "5. 今後の注目ポイント（次の決算で確認したいこと、株価が動くきっかけになりそうなこと）",
        "",
        "## 回答のルール",
        "- 可能であれば Web 検索で最新の決算短信・決算説明資料・ニュースを確認し、参照した情報の出所と時点を示してください。",
        "- 確認できなかった数字は推測で書かず、「不明」と書いてください。",
        "- 売買の推奨ではなく、判断材料の整理としてまとめてください。",
        "- 日本語で、見出しと箇条書きを使って簡潔に。",
    ]
    return "\n".join(lines)
