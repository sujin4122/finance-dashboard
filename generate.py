"""엘리스식 인과 대시보드 생성기.
FRED(API 키 불필요)에서 데이터를 받아 docs/index.html 한 장을 만든다.
임계값·비중 규칙은 아래 CONFIG만 고치면 된다.
"""
import io
import json
import os
import pathlib
import time
import urllib.request
from datetime import datetime, timezone, timedelta

import pandas as pd

# ───────────────────────── CONFIG ─────────────────────────
START = "2018-01-01"
SHOW_MONTHS = 36

# 지표별 임계값. higher_is_better=True면 값이 클수록 좋음.
# green_at: 이 값 이상(또는 이하)이면 초록 / red_at: 이 값 미만(또는 초과)이면 빨강, 사이는 노랑
INDICATORS = [
    # key, 단계, 이름, 단위, 좋은방향, 초록 기준, 빨강 기준, 점수반영
    ("real_wage",   1, "실질 시간당 임금 증가율", "% YoY, 3개월 평균", True,  1.0, 0.0, True),
    ("real_income", 1, "실질 소득 (이전소득 제외)", "% YoY, 3개월 평균", True,  1.0, 0.0, True),
    ("real_pce",    2, "실질 소비지출",         "% YoY, 3개월 평균", True,  2.0, 1.0, True),
    ("claims",      3, "신규 실업수당 청구(4주 평균)", "% YoY",        False, 10.0, 20.0, True),
    ("sahm",        3, "샴 룰 (침체 신호)",     "%p",               False, 0.3, 0.5, True),
    ("unrate",      3, "실업률 (확인용)",       "%",                False, None, None, False),
]
POINTS = {"green": 0.0, "yellow": 0.5, "red": 1.0}

# 경고 점수 → 주식 목표 비중. (점수 상한, 비중) 순서대로 검사
WEIGHT_RULES = [(1.0, 100), (2.5, 80), (99, 50)]
# 비중은 최근 2개월 점수 중 '덜 나쁜 쪽'을 기준으로 해서 한 달짜리 노이즈에 안 흔들리게 함
CONFIRM_MONTHS = 2
# ──────────────────────────────────────────────────────────

CACHE_DIR = pathlib.Path(__file__).parent / "cache"

SERIES = {
    "wage": "CES0500000008", "cpi": "CPIAUCSL", "income": "W875RX1",
    "pce": "PCEC96", "claims": "IC4WSA", "sahm": "SAHMREALTIME", "unrate": "UNRATE",
}


def _download(series_id: str) -> pd.Series:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:  # 공식 API (가장 안정적)
        url = ("https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={key}&file_type=json&observation_start={START}")
        with urllib.request.urlopen(url, timeout=60) as r:
            obs = json.loads(r.read().decode())["observations"]
        s = pd.Series([pd.to_numeric(o["value"], errors="coerce") for o in obs],
                      index=pd.to_datetime([o["date"] for o in obs]))
        return s.dropna()
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        df = pd.read_csv(io.StringIO(r.read().decode()))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]))
    return s.dropna()


def fetch(series_id: str) -> pd.Series:
    """재시도 후에도 실패하면 직전에 저장해 둔 캐시를 사용."""
    cache = CACHE_DIR / f"{series_id}.csv"
    last_err = None
    for attempt in range(1, 6):
        try:
            s = _download(series_id)
            if len(s) < 10:
                raise ValueError("데이터가 비정상적으로 짧음")
            CACHE_DIR.mkdir(exist_ok=True)
            s.to_csv(cache, header=["value"], index_label="date")
            return s
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[{series_id}] 시도 {attempt}/5 실패: {e}", flush=True)
            time.sleep(5 * attempt)
    if cache.exists():
        print(f"[{series_id}] 캐시 사용 (FRED 접속 실패)", flush=True)
        c = pd.read_csv(cache, index_col=0, parse_dates=True)
        return c["value"].dropna()
    raise RuntimeError(f"{series_id} 다운로드 실패, 캐시도 없음: {last_err}")


def monthly(s: pd.Series) -> pd.Series:
    return s.resample("MS").mean()


def build_frame(raw: dict) -> pd.DataFrame:
    m = {k: monthly(v) for k, v in raw.items()}
    real_wage = (m["wage"] / m["cpi"]).pct_change(12) * 100
    df = pd.DataFrame({
        "real_wage": real_wage.rolling(3).mean(),
        "real_income": (m["income"].pct_change(12) * 100).rolling(3).mean(),
        "real_pce": (m["pce"].pct_change(12) * 100).rolling(3).mean(),
        "claims": m["claims"].pct_change(12) * 100,
        "sahm": m["sahm"],
        "unrate": m["unrate"],
    })
    df = df.ffill(limit=2)  # 발표 시차로 최근 달이 비는 경우만 직전값으로 메움
    return df.dropna(how="all")


def signal(value, higher_better, green_at, red_at):
    if value is None or pd.isna(value) or green_at is None:
        return None
    if higher_better:
        return "green" if value >= green_at else ("red" if value < red_at else "yellow")
    return "green" if value < green_at else ("red" if value > red_at else "yellow")


def score_row(row) -> float:
    total = 0.0
    for key, _, _, _, hb, g, r, scored in INDICATORS:
        if scored:
            sig = signal(row.get(key), hb, g, r)
            if sig:
                total += POINTS[sig]
    return total


def weight_for(score: float) -> int:
    for cap, w in WEIGHT_RULES:
        if score < cap:
            return w
    return WEIGHT_RULES[-1][1]


COL = {"green": "#2e9e5b", "yellow": "#d9a400", "red": "#d64545", None: "#8a8f98"}
LABEL = {"green": "양호", "yellow": "주의", "red": "경고", None: "n/a"}


def sparkline(series: pd.Series, hb, g, r, w=240, h=64) -> str:
    vals = series.dropna().tolist()
    if len(vals) < 2:
        return ""
    lo = min(vals + [x for x in (g, r) if x is not None])
    hi = max(vals + [x for x in (g, r) if x is not None])
    span = (hi - lo) or 1
    pad = 4

    def y(v):
        return pad + (h - 2 * pad) * (1 - (v - lo) / span)

    pts = " ".join(f"{pad + (w - 2 * pad) * i / (len(vals) - 1):.1f},{y(v):.1f}"
                   for i, v in enumerate(vals))
    lines = ""
    for th, c in ((g, COL["green"]), (r, COL["red"])):
        if th is not None:
            lines += (f'<line x1="0" x2="{w}" y1="{y(th):.1f}" y2="{y(th):.1f}" '
                      f'stroke="{c}" stroke-dasharray="3 3" opacity=".55"/>')
    last_x = w - pad
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">'
            f'{lines}<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.8"/>'
            f'<circle cx="{last_x}" cy="{y(vals[-1]):.1f}" r="3.2" fill="currentColor"/></svg>')


def card(df, ind) -> str:
    key, _, name, unit, hb, g, r, scored = ind
    s = df[key].tail(SHOW_MONTHS)
    last = s.dropna()
    val = last.iloc[-1] if len(last) else None
    when = last.index[-1].strftime("%Y-%m") if len(last) else "-"
    sig = signal(val, hb, g, r)
    vtxt = "—" if val is None else f"{val:+.2f}" if key not in ("unrate", "sahm") else f"{val:.2f}"
    rule = ""
    if g is not None:
        rule = (f"초록 {'≥' if hb else '<'} {g:g} · 빨강 {'<' if hb else '>'} {r:g}")
    elif not scored:
        rule = "점수 미반영 · 참고용"
    return f"""
    <div class="card">
      <div class="top"><span class="name">{name}</span>
        <span class="pill" style="background:{COL[sig]}">{LABEL[sig]}</span></div>
      <div class="val">{vtxt}<small> {unit}</small></div>
      <div class="chart">{sparkline(s, hb, g, r)}</div>
      <div class="meta">기준월 {when} · {rule}</div>
    </div>"""


def render(df: pd.DataFrame) -> str:
    score_hist = df.apply(score_row, axis=1)
    recent = score_hist.tail(CONFIRM_MONTHS)
    confirmed = recent.min()
    now_score = score_hist.iloc[-1]
    weight = weight_for(confirmed)
    wcol = {100: COL["green"], 80: COL["yellow"]}.get(weight, COL["red"])

    # 최근 12개월 점수 막대
    bars = ""
    for d, sc in score_hist.tail(12).items():
        hgt = 6 + sc * 14
        c = COL["green"] if sc < 1 else COL["yellow"] if sc < 2.5 else COL["red"]
        bars += (f'<div class="bar"><i style="height:{hgt:.0f}px;background:{c}"></i>'
                 f'<span>{d.strftime("%m")}</span></div>')

    stages = {1: "① 원인 — 소득", 2: "② 중간 — 소비", 3: "③ 결과 — 고용·침체"}
    cols = ""
    for st in (1, 2, 3):
        cards = "".join(card(df, i) for i in INDICATORS if i[1] == st)
        cols += f'<section><h2>{stages[st]}</h2>{cards}</section>'

    kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    rules = " / ".join(f"{('<' + str(c)) if c < 99 else '그 이상'}점 → {w}%" for c, w in WEIGHT_RULES)

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>경제 인과 대시보드</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--tx:#1b1f24;--sub:#6b7280;--line:#e5e7eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1216;--card:#181c22;--tx:#e8eaed;--sub:#9aa1ab;--line:#2a2f37}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;padding:20px;max-width:1200px;margin:auto}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--sub);font-size:12px;margin-bottom:16px}}
.hero{{display:flex;flex-wrap:wrap;gap:16px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 22px;margin-bottom:18px}}
.big{{font-size:48px;font-weight:700;color:{wcol};line-height:1}}
.hero .lbl{{color:var(--sub);font-size:12px}}
.bars{{display:flex;gap:6px;align-items:flex-end;margin-left:auto;height:80px}}
.bar{{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;font-size:10px;color:var(--sub)}}
.bar i{{width:12px;border-radius:3px;display:block;margin-bottom:3px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}
@media (max-width:860px){{.grid{{grid-template-columns:1fr}}}}
h2{{font-size:13px;color:var(--sub);margin:0 0 10px}}
section{{position:relative}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}}
.top{{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}}
.name{{font-size:13px;font-weight:600}}
.pill{{color:#fff;font-size:11px;padding:2px 8px;border-radius:99px;white-space:nowrap}}
.val{{font-size:26px;font-weight:700;margin:6px 0 2px}} .val small{{font-size:11px;color:var(--sub);font-weight:400}}
.chart{{color:var(--tx)}} .meta{{font-size:11px;color:var(--sub);margin-top:4px}}
footer{{color:var(--sub);font-size:11px;margin-top:10px;line-height:1.6}}
</style></head><body>
<h1>경제를 읽는 기술 · 인과 대시보드</h1>
<div class="sub">실질 임금·소득 → 소비 → 청구건수·실업·침체 순서로 읽기 · 갱신 {kst}</div>
<div class="hero">
  <div><div class="lbl">권장 주식 목표 비중</div><div class="big">{weight}%</div></div>
  <div><div class="lbl">경고 점수(최근 {CONFIRM_MONTHS}개월 중 낮은 쪽)</div>
       <div style="font-size:24px;font-weight:700">{confirmed:g} <small style="font-size:12px;color:var(--sub)">/ 현재월 {now_score:g}</small></div>
       <div class="lbl">경고=1점, 주의=0.5점 · {rules}</div></div>
  <div class="bars" title="최근 12개월 경고 점수">{bars}</div>
</div>
<div class="grid">{cols}</div>
<footer>데이터: FRED(St. Louis Fed). 점선은 초록/빨강 기준선. 월간 지표는 발표 시차와 사후 수정이 있습니다.
참고용 도구이며 투자 판단과 책임은 본인에게 있습니다. 임계값과 비중 규칙은 generate.py 상단 CONFIG에서 수정합니다.</footer>
</body></html>"""


def main():
    raw = {k: fetch(v) for k, v in SERIES.items()}
    df = build_frame(raw)
    out = pathlib.Path(__file__).parent / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(render(df), encoding="utf-8")
    print("OK", df.index[-1].date(), "score", score_row(df.iloc[-1]))


if __name__ == "__main__":
    main()
