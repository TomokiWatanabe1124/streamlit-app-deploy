"""東証 夕方チェック — 帰宅後に当日の相場をまとめて確認する Streamlit アプリ。"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from stockapp import jpx, market, tdnet
from stockapp.watchlist import load_watchlist, to_yahoo

st.set_page_config(page_title="東証 夕方チェック", page_icon="📈", layout="wide")

JST = dt.timezone(dt.timedelta(hours=9))
WEEKDAYS = "月火水木金土日"
RANK_N = 20
ALERT_PCT = 5.0


def jp_date(d: dt.date) -> str:
    return f"{d:%Y-%m-%d}（{WEEKDAYS[d.weekday()]}）"


# ---------------------------------------------------------------------------
# データ取得（キャッシュ付き）
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 24, show_spinner="上場銘柄一覧を取得中…")
def get_listing() -> pd.DataFrame:
    return jpx.fetch_listing()


@st.cache_data(ttl=60 * 30, show_spinner="全銘柄の相場を取得中…")
def get_market(codes: tuple[str, ...]) -> pd.DataFrame:
    return market.fetch_market(list(codes))


@st.cache_data(ttl=60 * 30, show_spinner="チャート用データを取得中…")
def get_history(symbols: tuple[str, ...], period: str = "6mo") -> dict[str, pd.DataFrame]:
    return market.fetch_history(list(symbols), period)


@st.cache_data(ttl=60 * 30, show_spinner="適時開示を取得中…")
def get_disclosures(day: dt.date) -> pd.DataFrame:
    return tdnet.fetch_day(day)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="決算発表予定を取得中…")
def get_earnings() -> pd.DataFrame:
    return jpx.fetch_earnings_schedule()


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def get_yahoo_earnings_date(symbol: str) -> dt.date | None:
    """JPX の予定表に載っていない銘柄の補助として、Yahoo の決算予定日を使う。"""
    try:
        cal = yf.Ticker(symbol).calendar or {}
        dates = cal.get("Earnings Date") or []
        today = dt.datetime.now(JST).date()
        future = sorted(d for d in dates if d >= today)
        return future[0] if future else None
    except Exception:
        return None


def safe(fn, *args, label: str):
    """取得に失敗してもアプリ全体を止めず、警告を出して None を返す。"""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        st.warning(f"{label}の取得に失敗しました: {e}")
        return None


# ---------------------------------------------------------------------------
# 表示用の小物
# ---------------------------------------------------------------------------

def fmt_price(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:,.1f}" if abs(v) < 1000 and v != int(v) else f"{v:,.0f}"


def color_change(v):
    if pd.isna(v) or v == 0:
        return ""
    return "color: #d32f2f" if v > 0 else "color: #1565c0"  # 日本式: 上昇=赤, 下落=青


def highlight_row(row: pd.Series):
    if str(row.get("注目", "")):
        return ["background-color: rgba(255, 193, 7, 0.18)"] * len(row)
    return [""] * len(row)


def price_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "コード": df["code"],
            "銘柄名": df["name"],
            "市場": df.get("market", ""),
            "終値": df["close"],
            "前日比": df["change"],
            "前日比%": df["change_pct"],
            "出来高": df["volume"],
            "出来高倍率": df["volume_ratio"],
            "S高/S安": df["limit"],
        }
    )
    return out


PRICE_COLUMN_CONFIG = {
    "終値": st.column_config.NumberColumn(format="%.1f"),
    "前日比": st.column_config.NumberColumn(format="%+.1f"),
    "前日比%": st.column_config.NumberColumn(format="%+.2f%%"),
    "出来高": st.column_config.NumberColumn(format="localized"),
    "出来高倍率": st.column_config.NumberColumn(help="当日出来高 ÷ 3か月平均出来高", format="%.1f倍"),
}


def show_price_table(df: pd.DataFrame, height: int | None = None, highlight: bool = True):
    styled = df.style.map(color_change, subset=[c for c in ("前日比", "前日比%") if c in df.columns])
    if highlight:
        styled = styled.apply(highlight_row, axis=1)
    kwargs = {"height": height} if height else {}
    st.dataframe(styled, hide_index=True, width="stretch", column_config=PRICE_COLUMN_CONFIG, **kwargs)


def candle_chart(d: pd.DataFrame, title: str) -> go.Figure:
    d = d.copy()
    d["MA25"] = d["Close"].rolling(25).mean()
    d["MA75"] = d["Close"].rolling(75).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="日足",
            increasing_line_color="#d32f2f", decreasing_line_color="#1565c0",
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=d.index, y=d["MA25"], name="25日線", line=dict(width=1.2, color="#ef6c00")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MA75"], name="75日線", line=dict(width=1.2, color="#2e7d32")), row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="出来高", marker_color="#90a4ae"), row=2, col=1)
    fig.update_layout(
        title=title, height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

st.title("📈 東証 夕方チェック")

with st.sidebar:
    st.header("設定")
    if st.button("🔄 最新データに更新", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("データは30分間キャッシュされます。")
    st.divider()
    st.markdown(
        "**ウォッチ銘柄の追加・削除**\n\n"
        "リポジトリの `watchlist.csv` を編集してください。GitHub の画面で編集してコミットすると、"
        "アプリに自動で反映されます。"
    )

watch = load_watchlist()
listing = safe(get_listing, label="上場銘柄一覧")
if listing is None or listing.empty:
    st.error("上場銘柄一覧が取得できないため、表示できません。時間をおいて再読み込みしてください。")
    st.stop()

mkt = safe(get_market, tuple(listing["code"]), label="全銘柄の相場")
if mkt is None or mkt.empty:
    st.error("相場データが取得できませんでした。時間をおいて「最新データに更新」を押してください。")
    st.stop()
mkt = mkt.merge(listing, on="code", how="left")
market_day: dt.date = mkt["date"].max()

now = dt.datetime.now(JST)
st.caption(
    f"対象: **{jp_date(market_day)}の相場**　｜　取得時刻 {now:%Y-%m-%d %H:%M}"
    + ("　｜　⚠️ 取引時間中のため、まだ終値ではありません" if now.date() == market_day and now.weekday() < 5 and (9 <= now.hour < 15 or (now.hour == 15 and now.minute < 30)) else "")
)

# 適時開示: 相場の営業日の分（大引け後の開示を含む）。今日が休日なら今日の分も足す。
disc_days = [market_day] + ([now.date()] if now.date() > market_day else [])
disc_frames = [safe(get_disclosures, d, label=f"{d:%m/%d} の適時開示") for d in disc_days]
disc = pd.concat([f.assign(day=d) for f, d in zip(disc_frames, disc_days) if f is not None], ignore_index=True) \
    if any(f is not None for f in disc_frames) else pd.DataFrame(columns=tdnet.COLUMNS + ["day"])

earnings = safe(get_earnings, label="決算発表予定")

# ウォッチ銘柄の相場
w = watch.merge(mkt.drop(columns=["name"]), on="code", how="left")
w["name"] = w["name"].where(w["name"] != "", w["code"].map(dict(zip(listing["code"], listing["name"]))))


def next_earnings(code: str) -> tuple[dt.date | None, str]:
    today = now.date()
    if earnings is not None and not earnings.empty:
        hit = earnings[(earnings["code"] == code) & (earnings["date"] >= today)]
        if not hit.empty:
            r = hit.iloc[0]
            return r["date"], f"JPX（{r['kind']}）"
    d = get_yahoo_earnings_date(to_yahoo(code))
    return (d, "Yahoo（参考・推定の場合あり）") if d else (None, "")


def watch_flags(row: pd.Series) -> str:
    flags = []
    if pd.notna(row.get("change_pct")) and abs(row["change_pct"]) >= ALERT_PCT:
        flags.append("📈 +5%以上" if row["change_pct"] > 0 else "📉 -5%以下")
    if row.get("limit"):
        flags.append(f"🚨 {row['limit']}")
    titles = disc.loc[disc["code"] == row["code"], "category"]
    if (titles == "業績修正").any():
        flags.append("📝 業績修正")
    return " ".join(flags)


w["注目"] = w.apply(watch_flags, axis=1)

# ---------------------------------------------------------------------------
# タブ
# ---------------------------------------------------------------------------

tab_sum, tab_watch, tab_rank, tab_disc, tab_earn = st.tabs(
    ["📊 サマリー", "⭐ ウォッチリスト", "🏆 ランキング", "📰 適時開示", "📅 決算スケジュール"]
)

# --- サマリー -------------------------------------------------------------
with tab_sum:
    st.subheader("今日のチェックポイント")
    alerts = w[w["注目"] != ""]
    if alerts.empty:
        st.success("ウォッチ銘柄に大きな動き（±5%、ストップ高・安、業績修正）はありませんでした。")
    else:
        for _, r in alerts.iterrows():
            st.warning(f"**{r['code']} {r['name']}**　{r['注目']}　（前日比 {r['change_pct']:+.2f}%）")

    soon = []
    for _, r in watch.iterrows():
        d, _src = next_earnings(r["code"])
        if d and (d - now.date()).days <= 14:
            soon.append(f"{r['code']} {r['name'] or r['code']}：{jp_date(d)}")
    if soon:
        st.info("📅 2週間以内に決算発表予定のウォッチ銘柄\n\n" + "\n".join(f"- {s}" for s in soon))

    st.subheader("指数")
    idx_hist = safe(get_history, tuple(s for _, s, _ in market.INDICES), "3mo", label="指数")
    cols = st.columns(len(market.INDICES))
    for col, (label, sym, note) in zip(cols, market.INDICES):
        d = (idx_hist or {}).get(sym)
        if d is None or len(d) < 2:
            col.metric(label, "-")
            continue
        last, prev = d["Close"].iloc[-1], d["Close"].iloc[-2]
        col.metric(
            label, f"{last:,.2f}", f"{last - prev:+,.2f} ({(last / prev - 1) * 100:+.2f}%)",
            delta_color="off" if sym == "JPY=X" else "normal", help=note or None,
        )
        col.caption(f"{d.index[-1]:%m/%d} 時点")

    st.subheader("相場全体の雰囲気")
    up = int((mkt["change_pct"] > 0).sum())
    down = int((mkt["change_pct"] < 0).sum())
    flat = int((mkt["change_pct"] == 0).sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("値上がり銘柄", f"{up:,}")
    c2.metric("値下がり銘柄", f"{down:,}")
    c3.metric("変わらず", f"{flat:,}")
    c4.metric("ストップ高（到達含む）", f"{int(mkt['limit'].str.startswith('S高').sum()):,}")
    c5.metric("ストップ安（到達含む）", f"{int(mkt['limit'].str.startswith('S安').sum()):,}")
    st.progress(up / max(up + down, 1), text=f"値上がり比率 {up / max(up + down, 1) * 100:.0f}%")

    by_market = (
        mkt.groupby("market")["change_pct"]
        .agg(値上がり=lambda s: (s > 0).sum(), 値下がり=lambda s: (s < 0).sum(), 平均騰落率="mean")
        .reindex(["プライム", "スタンダード", "グロース"])
        .rename_axis("市場")
    )
    sector = mkt[mkt["market"] == "プライム"].groupby("sector")["change_pct"].mean().sort_values()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**市場別**")
        st.dataframe(by_market, column_config={"平均騰落率": st.column_config.NumberColumn(format="%+.2f%%")})
    with c2:
        st.markdown("**業種別 平均騰落率（プライム）**")
        fig = go.Figure(go.Bar(
            x=sector.values, y=sector.index, orientation="h",
            marker_color=["#d32f2f" if v > 0 else "#1565c0" for v in sector.values],
            text=[f"{v:+.2f}%" for v in sector.values], textposition="outside",
        ))
        fig.update_layout(height=760, margin=dict(l=10, r=40, t=10, b=10), xaxis_ticksuffix="%")
        st.plotly_chart(fig, width="stretch")

# --- ウォッチリスト ---------------------------------------------------------
with tab_watch:
    st.subheader(f"ウォッチ銘柄（{len(watch)}銘柄）")
    rows = []
    for _, r in w.iterrows():
        d, src = next_earnings(r["code"])
        rows.append({"次回決算": jp_date(d) if d else "未定", "予定日の出所": src})
    wt = pd.concat([price_table(w), pd.DataFrame(rows), w[["注目", "memo"]].rename(columns={"memo": "メモ"})], axis=1)
    wt.insert(2, "注目", wt.pop("注目"))
    show_price_table(wt)
    missing = w[w["close"].isna()]
    if not missing.empty:
        st.caption("相場が取得できなかった銘柄: " + "、".join(missing["code"] + " " + missing["name"]))

    st.markdown("**直近の適時開示（ウォッチ銘柄）**")
    wd = disc[disc["code"].isin(watch["code"])]
    if wd.empty:
        st.caption("対象日の開示はありません。")
    else:
        st.dataframe(
            wd[["day", "time", "code", "name", "category", "title", "url"]]
            .rename(columns={"day": "日付", "time": "時刻", "code": "コード", "name": "銘柄名", "category": "分類", "title": "表題", "url": "PDF"}),
            hide_index=True, width="stretch",
            column_config={"PDF": st.column_config.LinkColumn(display_text="開く")},
        )

    st.markdown("**チャート**")
    choice = st.selectbox(
        "銘柄", w["code"].tolist(), format_func=lambda c: f"{c} {w.loc[w['code'] == c, 'name'].iloc[0]}",
    )
    period = st.radio("期間", ["3mo", "6mo", "1y", "2y"], index=1, horizontal=True,
                      format_func={"3mo": "3か月", "6mo": "6か月", "1y": "1年", "2y": "2年"}.get)
    hist = safe(get_history, (to_yahoo(choice),), period, label="チャート")
    h = (hist or {}).get(to_yahoo(choice))
    if h is not None:
        st.plotly_chart(candle_chart(h, f"{choice} {w.loc[w['code'] == choice, 'name'].iloc[0]}"), width="stretch")

# --- ランキング -------------------------------------------------------------
with tab_rank:
    st.subheader(f"東証全体ランキング（{jp_date(market_day)}、{len(mkt):,}銘柄）")
    c1, c2 = st.columns([2, 1])
    markets = c1.multiselect("市場", ["プライム", "スタンダード", "グロース"], default=["プライム", "スタンダード", "グロース"])
    min_vol = c2.number_input("最低出来高（株）", min_value=0, value=0, step=10_000,
                              help="出来高が極端に少ない銘柄を除きたいときに使います")
    base = mkt[mkt["market"].isin(markets) & (mkt["volume"] >= min_vol)]

    up_tab, down_tab = st.tabs([f"📈 値上がり率 上位{RANK_N}", f"📉 値下がり率 上位{RANK_N}"])
    with up_tab:
        show_price_table(price_table(base.nlargest(RANK_N, "change_pct")), height=740, highlight=False)
    with down_tab:
        show_price_table(price_table(base.nsmallest(RANK_N, "change_pct")), height=740, highlight=False)

    st.markdown("**ストップ高・ストップ安**")
    lim = base[base["limit"] != ""].sort_values(["limit", "change_pct"], ascending=[True, False])
    if lim.empty:
        st.caption("該当銘柄はありません。")
    else:
        show_price_table(price_table(lim), highlight=False)
    st.caption("ストップ高・安は、前日終値と東証の通常の制限値幅から計算しています。値幅の拡大措置などの例外は考慮していません。")

# --- 適時開示 ---------------------------------------------------------------
with tab_disc:
    st.subheader("適時開示（TDnet）")
    c1, c2, c3 = st.columns([1, 1, 2])
    day = c1.date_input("日付", value=market_day, max_value=now.date(), min_value=now.date() - dt.timedelta(days=30))
    cat = c2.selectbox("分類", ["業績修正", "配当", "決算", "すべて"])
    kw = c3.text_input("キーワード（表題・銘柄名）", "")
    only_watch = st.checkbox("ウォッチ銘柄のみ", value=False)

    dd = disc[disc["day"] == day] if day in disc_days else safe(get_disclosures, day, label="適時開示")
    if dd is None or dd.empty:
        st.caption("この日の開示はありません（休日、またはまだ公開されていません）。")
    else:
        view = dd if cat == "すべて" else dd[dd["category"] == cat]
        if kw:
            view = view[view["title"].str.contains(kw, regex=False) | view["name"].str.contains(kw, regex=False)]
        if only_watch:
            view = view[view["code"].isin(watch["code"])]
        view = view.merge(mkt[["code", "change_pct"]], on="code", how="left")
        st.caption(f"{jp_date(day)}：全{len(dd):,}件中 {len(view):,}件を表示")
        st.dataframe(
            view[["time", "code", "name", "category", "title", "change_pct", "url"]].rename(
                columns={"time": "時刻", "code": "コード", "name": "銘柄名", "category": "分類",
                         "title": "表題", "change_pct": "当日前日比%", "url": "PDF"}
            ),
            hide_index=True, width="stretch", height=min(600, 35 * (len(view) + 1) + 3),
            column_config={
                "PDF": st.column_config.LinkColumn(display_text="開く"),
                "当日前日比%": st.column_config.NumberColumn(format="%+.2f%%", help=f"{jp_date(market_day)}の騰落率"),
            },
        )
        st.caption("15:30 以降の開示は、翌営業日の株価に反映されます。")

# --- 決算スケジュール -------------------------------------------------------
with tab_earn:
    st.subheader("決算発表スケジュール（東証全体）")
    if earnings is None or earnings.empty:
        st.info("決算発表予定を取得できませんでした。")
    else:
        today = now.date()
        upcoming = earnings[earnings["date"] >= today].merge(listing[["code", "market"]], on="code", how="left", suffixes=("_jpx", ""))
        upcoming["market"] = upcoming["market"].fillna(upcoming["market_jpx"])

        horizon = upcoming[upcoming["date"] <= today + dt.timedelta(days=28)]
        counts = horizon.groupby("date").size()
        if not counts.empty:
            fig = go.Figure(go.Bar(x=[jp_date(d) for d in counts.index], y=counts.values, marker_color="#5c6bc0",
                                   text=counts.values, textposition="outside"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="今後4週間の発表予定件数")
            st.plotly_chart(fig, width="stretch")

        c1, c2, c3 = st.columns([1, 2, 2])
        dates = sorted(upcoming["date"].unique())
        if dates:
            sel = c1.selectbox("発表日", dates, format_func=jp_date)
            mk = c2.multiselect("市場", sorted(upcoming["market"].dropna().unique()), key="earn_market", placeholder="すべて")
            kw2 = c3.text_input("銘柄名・コード", "", key="earn_kw")
            ev = upcoming[upcoming["date"] == sel]
            if mk:
                ev = ev[ev["market"].isin(mk)]
            if kw2:
                ev = ev[ev["name"].str.contains(kw2, regex=False) | ev["code"].str.contains(kw2, regex=False)]
            ev = ev.assign(ウォッチ=ev["code"].isin(watch["code"]).map({True: "⭐", False: ""}))
            st.dataframe(
                ev[["ウォッチ", "code", "name", "market", "sector", "kind", "period"]].rename(
                    columns={"code": "コード", "name": "会社名", "market": "市場", "sector": "業種", "kind": "種別", "period": "決算期末"}
                ),
                hide_index=True, width="stretch",
            )
        latest_known = earnings["date"].max()
        st.caption(
            f"出所: 日本取引所グループ「決算発表予定日」。現在 {jp_date(latest_known)} までの予定が公開されています。"
            "JPX は決算期末の月ごとに予定を公表するため、先の日付ほど件数が少なく表示されます。"
        )
