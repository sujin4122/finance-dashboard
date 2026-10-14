"""엘리스식 인과 대시보드 생성기. (개인·비상업용)
FRED에서 데이터를 받아 docs/index.html 한 장을 만든다.
임계값·비중 규칙은 아래 CONFIG만 고치면 된다.
"""
import io
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

import pandas as pd

# ───────────────────────── CONFIG ─────────────────────────
START = "2018-01-01"
SHOW_MONTHS = 36       # 카드 안 작은 차트 기간
CHART_MONTHS = 48      # 하단 겹친 차트 기간

# key, 단계, 이름, 단위, 좋은방향(True=높을수록 좋음), 초록 기준, 빨강 기준, 점수반영
INDICATORS = [
    ("real_wage",   1, "실질 시간당 임금 증가율", "% YoY, 3개월 평균", True,  1.0, 0.0, True),
    ("real_income", 1, "실질 소득 (이전소득 제외)", "% YoY, 3개월 평균", True,  1.0, 0.0, True),
    ("real_pce",    2, "실질 소비지출",         "% YoY, 3개월 평균", True,  2.0, 1.0, True),
    ("claims",      3, "신규 실업수당 청구(4주 평균)", "% YoY",        False, 10.0, 20.0, True),
    ("sahm",        3, "샴 룰 (침체 신호)",     "%p",               False, 0.3, 0.5, True),
    ("unrate",      3, "실업률 (확인용)",       "%",                False, None, None, False),
]
# 지표 하나당 점수: 양호 0점 / 주의 0.5점 / 경고 1점  →  5개 지표 합계 0~5점
POINTS = {"green": 0.0, "yellow": 0.5, "red": 1.0}

# (점수 상한 미만, 주식 비중 %, 상태 이름). 비중은 '전체 자산 대비 주식 상한'이며 나머지는 채권·현금.
# 보수형: 항상 최소 30%는 채권·현금으로 보유, 주식은 최대 70%.
ZONES = [
    (1.0, 70, "안정"),
    (2.0, 60, "둔화"),
    (3.0, 50, "경계"),
    (4.0, 40, "위험"),
    (99.0, 30, "침체급"),
]
CONFIRM_MONTHS = 2     # 최근 N개월 점수 중 낮은 쪽을 '확정 점수'로 사용
# ──────────────────────────────────────────────────────────

SERIES = {
    "wage": "AHETPI", "cpi": "CPIAUCSL", "income": "W875RX1",
    "pce": "PCEC96", "claims": "IC4WSA", "sahm": "SAHMREALTIME", "unrate": "UNRATE",
}


# ───────────────────────── 데이터 ─────────────────────────
def _download(series_id: str) -> pd.Series:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
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
    """최대 5회 재시도. 데이터를 파일로 저장(캐시)하지 않는다(FRED 약관).
    끝내 실패하면 오류로 종료하고, 이 경우 기존에 게시된 페이지가 그대로 유지된다."""
    last_err = None
    for attempt in range(1, 6):
        try:
            s = _download(series_id)
            if len(s) < 10:
                raise ValueError("데이터가 비정상적으로 짧음")
            return s
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:300]
            if e.code in (400, 404):  # 키 오류 또는 잘못된 시리즈 ID: 재시도 무의미
                raise RuntimeError(f"[{series_id}] HTTP {e.code}: {body}") from e
            last_err = e
            print(f"[{series_id}] 시도 {attempt}/5 실패: HTTP {e.code}", flush=True)
            time.sleep(5 * attempt)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[{series_id}] 시도 {attempt}/5 실패: {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"{series_id} 다운로드 실패: {last_err}")


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
    df = df.ffill(limit=2)  # 발표 시차·휴지 구간으로 비는 달만 직전값으로 메움
    return df.dropna(how="all")


# ───────────────────────── 판정 ─────────────────────────
def signal(value, higher_better, green_at, red_at):
    if value is None or pd.isna(value) or green_at is None:
        return None
    if higher_better:
        return "green" if value >= green_at else ("red" if value < red_at else "yellow")
    return "green" if value < green_at else ("red" if value > red_at else "yellow")


def signals_of(row) -> dict:
    return {i[0]: signal(row.get(i[0]), i[4], i[5], i[6]) for i in INDICATORS if i[7]}


def score_row(row) -> float:
    return sum(POINTS[s] for s in signals_of(row).values() if s)


def zone_index(score: float) -> int:
    for i, (cap, _, _) in enumerate(ZONES):
        if score < cap:
            return i
    return len(ZONES) - 1


# ───────────────────────── 그리기 ─────────────────────────
COL = {"green": "#2e9e5b", "yellow": "#d9a400", "red": "#d64545", None: "#8a8f98"}
LABEL = {"green": "양호", "yellow": "주의", "red": "경고", None: "n/a"}
ZCOL = ["#2e9e5b", "#d9a400", "#e8822a", "#d64545", "#9b1c1c"]


def sparkline(series, g, r, w=240, h=64) -> str:
    vals = series.dropna().tolist()
    if len(vals) < 2:
        return ""
    ths = [x for x in (g, r) if x is not None]
    lo, hi = min(vals + ths), max(vals + ths)
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
                      f'stroke="{c}" stroke-width="1.5" stroke-dasharray="4 3" opacity=".8"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">'
            f'{lines}<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.8"/>'
            f'<circle cx="{w - pad}" cy="{y(vals[-1]):.1f}" r="3.2" fill="currentColor"/></svg>')


def card(df, ind) -> str:
    key, _, name, unit, hb, g, r, scored = ind
    s = df[key].tail(SHOW_MONTHS)
    last = s.dropna()
    val = last.iloc[-1] if len(last) else None
    when = last.index[-1].strftime("%Y-%m") if len(last) else "-"
    sig = signal(val, hb, g, r)
    if val is None:
        vtxt = "—"
    elif key in ("unrate", "sahm"):
        vtxt = f"{val:.2f}"
    else:
        vtxt = f"{val:+.2f}"
    if g is not None:
        gtxt = f"{'≥' if hb else '<'} {g:g}"
        rtxt = f"{'<' if hb else '>'} {r:g}"
        rule = (f'<span style="color:{COL["green"]}">┄ 초록선 {gtxt}</span> · '
                f'<span style="color:{COL["red"]}">┄ 빨강선 {rtxt}</span>')
    else:
        rule = "점수 미반영 · 참고용"
    return f"""
    <div class="card">
      <div class="top"><span class="name">{name}</span>
        <span class="pill" style="background:{COL[sig]}">{LABEL[sig]}</span></div>
      <div class="val">{vtxt}<small> {unit}</small></div>
      <div class="chart">{sparkline(s, g, r)}</div>
      <div class="meta">기준월 {when} · {rule}</div>
    </div>"""


def combo_chart(df) -> str:
    """원인→결과 시차를 보기 위한 겹친 차트. 각 선을 표준화하고 '아래로 = 나쁨'으로 통일."""
    spec = [("real_wage", "① 실질 임금", "#2b7de9", 1),
            ("real_income", "① 실질 소득", "#14a3a3", 1),
            ("real_pce", "② 실질 소비", "#8e5bd1", 1),
            ("claims", "③ 청구건수 (뒤집음)", "#e8822a", -1)]
    d = df[[k for k, *_ in spec]].tail(CHART_MONTHS)
    zs = {}
    for k, _, _, sign in spec:
        s = d[k]
        if s.notna().sum() < 6 or not s.std():
            continue
        zs[k] = ((s - s.mean()) / s.std()) * sign
    if not zs:
        return ""
    allv = pd.concat(zs.values()).dropna()
    ymin, ymax = allv.min() - 0.3, allv.max() + 0.3
    W, H, pl, pr, pt, pb = 900, 300, 34, 14, 14, 28
    n = len(d)

    def X(i):
        return pl + (W - pl - pr) * i / (n - 1)

    def Y(v):
        return pt + (H - pt - pb) * (1 - (v - ymin) / (ymax - ymin))

    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block">']
    out.append(f'<line x1="{pl}" x2="{W - pr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" '
               f'stroke="currentColor" opacity=".35" stroke-dasharray="4 4"/>')
    out.append(f'<text x="2" y="{pt + 10}" font-size="11" fill="currentColor" opacity=".7">좋음 ↑</text>')
    out.append(f'<text x="2" y="{H - pb - 2}" font-size="11" fill="currentColor" opacity=".7">나쁨 ↓</text>')
    for i, dt in enumerate(d.index):
        if dt.month in (1, 7):
            out.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{pt}" y2="{H - pb}" '
                       f'stroke="currentColor" opacity=".1"/>')
            out.append(f'<text x="{X(i):.1f}" y="{H - 8}" font-size="11" text-anchor="middle" '
                       f'fill="currentColor" opacity=".7">{dt.strftime("%y-%m")}</text>')
    legend = ""
    for k, name, color, _ in spec:
        if k not in zs:
            continue
        pts = [(X(i), Y(v)) for i, v in enumerate(zs[k].values) if pd.notna(v)]
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
                   f'fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
        out.append(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" fill="{color}"/>')
        legend += f'<span><i style="background:{color}"></i>{name}</span>'
    out.append("</svg>")
    return f'<div class="legend">{legend}</div>' + "".join(out)


def render(df: pd.DataFrame) -> str:
    score_hist = df.apply(score_row, axis=1)
    confirmed = score_hist.tail(CONFIRM_MONTHS).min()
    now_score = score_hist.iloc[-1]
    prev_score = score_hist.iloc[-2] if len(score_hist) > 1 else now_score
    zi = zone_index(confirmed)
    _, weight, zname = ZONES[zi]
    zcol = ZCOL[zi]

    sigs = signals_of(df.iloc[-1])
    n_red = sum(1 for s in sigs.values() if s == "red")
    n_yel = sum(1 for s in sigs.values() if s == "yellow")
    n_grn = sum(1 for s in sigs.values() if s == "green")
    total = len(sigs)

    # 게이지 (0~5점)
    maxscore = float(total)
    seg = "".join(
        f'<div style="background:{ZCOL[i]}"><span>{nm}</span></div>'
        for i, (_, _, nm) in enumerate(ZONES))
    pos = min(max(confirmed / maxscore, 0), 1) * 100
    ticks = "".join(f"<em>{k}</em>" for k in range(int(maxscore) + 1))

    # 비중표
    rows = ""
    lo = 0.0
    for i, (cap, w, nm) in enumerate(ZONES):
        rng = f"{lo:g}점 이상 ~ {cap:g}점 미만" if cap < 99 else f"{lo:g}점 이상"
        cur = ' class="cur"' if i == zi else ""
        rows += (f'<tr{cur}><td><b style="color:{ZCOL[i]}">●</b> {nm}</td><td>{rng}</td>'
                 f'<td><b>{w}%</b></td><td>{100 - w}%</td></tr>')
        lo = cap

    # 최근 12개월 막대
    c0, c2 = ZONES[0][0], ZONES[2][0]
    bars = ""
    for d, sc in score_hist.tail(12).items():
        c = COL["green"] if sc < c0 else COL["yellow"] if sc < c2 else COL["red"]
        bars += (f'<div class="bar"><b>{sc:g}</b><i style="height:{6 + sc * 14:.0f}px;background:{c}"></i>'
                 f'<span>{d.strftime("%m")}월</span></div>')

    stages = {1: "① 원인 — 소득", 2: "② 중간 — 소비", 3: "③ 결과 — 고용·침체"}
    cols = "".join(
        f'<section><h2>{stages[st]}</h2>{"".join(card(df, i) for i in INDICATORS if i[1] == st)}</section>'
        for st in (1, 2, 3))

    kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow, noarchive">
<title>경제 인과 대시보드</title>
<style>{CSS}</style></head><body>
<h1>경제를 읽는 기술 · 인과 대시보드</h1>
<div class="sub">실질 임금·소득 → 소비 → 청구건수·실업·침체 순서로 읽기 · 갱신 {kst}</div>

<div class="panel hero">
  <div class="hrow">
    <div><div class="lbl">권장 주식 비중 (전체 자산 기준 상한)</div>
         <div class="big" style="color:{zcol}">{weight}%</div>
         <div class="lbl">나머지 {100 - weight}%는 채권·현금</div></div>
    <div><div class="lbl">경기 경고 점수 (최근 {CONFIRM_MONTHS}개월 중 낮은 쪽)</div>
         <div class="score"><span style="color:{zcol}">{confirmed:g}</span> <small>/ {maxscore:g}점 · {zname}</small></div>
         <div class="lbl">이번 달 {now_score:g}점 · 지난달 {prev_score:g}점<br>
         이번 달 지표 {total}개 중 <b style="color:{COL['red']}">경고 {n_red}</b> ·
         <b style="color:{COL['yellow']}">주의 {n_yel}</b> · <b style="color:{COL['green']}">양호 {n_grn}</b></div></div>
  </div>
  <div class="gauge"><div class="segs">{seg}</div>
    <div class="mark" style="left:{pos:.1f}%"><b>▼</b></div>
    <div class="ticks">{ticks}</div></div>
  <div class="how">점수 계산: 지표마다 <b>양호 0점 · 주의 0.5점 · 경고 1점</b>을 주고 {total}개를 더합니다.
  전부 양호면 0점, 전부 경고면 {maxscore:g}점입니다. 예) 경고 1개 + 주의 1개 = 1.5점.</div>
</div>

<div class="two">
  <div class="panel"><h3>점수별 주식 비중표</h3>
    <table><tr><th>상태</th><th>점수</th><th>주식</th><th>채권·현금</th></tr>{rows}</table>
    <div class="note">70%가 상한입니다(30%는 항상 채권·현금 유지). 100%는 이 표가 아니라,
    코로나급 급락 때 예비자금을 쓰는 <b>본인 판단</b>의 영역입니다.</div></div>
  <div class="panel"><h3>최근 12개월 경고 점수</h3>
    <div class="bars">{bars}</div>
    <div class="note"><span style="color:{COL['green']}">■ 초록</span> 안정({c0:g}점 미만) ·
    <span style="color:{COL['yellow']}">■ 노랑</span> 둔화·경계({c0:g}~{c2:g}점) ·
    <span style="color:{COL['red']}">■ 빨강</span> 위험 이상({c2:g}점 이상).
    막대 위 숫자가 그 달 점수이며, 몇 달째 오르는 중인지가 핵심입니다.</div></div>
</div>

<div class="grid">{cols}</div>

<div class="panel" style="margin-top:6px"><h3>한눈에 보기 — 원인이 결과를 앞서는지 확인</h3>
  {combo_chart(df)}
  <div class="note">각 선을 같은 눈금으로 맞췄고(최근 {CHART_MONTHS}개월 평균=점선), 모두 <b>아래로 내려갈수록 나쁨</b>입니다.
  ① 소득선이 먼저 꺾이고 → ② 소비선 → ③ 청구건수선 순서로 따라 내려가면 엘리스식 경기 둔화 신호입니다.
  선들의 현재 위치(끝 점)가 점선보다 아래에 몰려 있는지 보세요.</div></div>

<details class="panel"><summary>대시보드 읽는 법</summary>
<ul>
<li><b>초록 점선·빨강 점선</b>: 카드 차트의 판단 기준선입니다. 실질 임금·소득·소비처럼 높을수록 좋은 지표는 초록선 위면 양호, 빨강선 아래면 경고입니다. 청구건수·샴 룰은 낮을수록 좋아서 초록선 아래면 양호, 빨강선 위면 경고입니다. 두 선 사이는 주의입니다.</li>
<li><b>샴 룰</b>: 실업률 3개월 평균이 직전 12개월 최저 대비 0.5%p 이상 오르면 침체 신호로 봅니다. 실업이 이미 오른 뒤에 나타나는 확인용 신호라 늦게 울리고, 과거 규칙일 뿐 법칙은 아닙니다.</li>
<li><b>점수 해석</b>: 1점 미만은 대체로 안정, 1~2점은 일부 지표가 나빠지는 둔화 초기, 3점 이상은 여러 지표가 동시에 나쁜 구간입니다. 한 달 점수보다 방향(상승 중인지)이 더 중요합니다.</li>
</ul></details>

<footer>
<div class="fred"><b>This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.</b><br>
데이터 출처: U.S. Bureau of Labor Statistics, U.S. Bureau of Economic Analysis, U.S. Department of Labor 등 via FRED®, Federal Reserve Bank of St. Louis (fred.stlouisfed.org).
이 페이지를 이용하면 <a href="https://fred.stlouisfed.org/legal/terms_of_use.html">FRED® 이용약관(API 약관 포함)</a>에 동의한 것으로 봅니다.</div>
월간 지표는 발표 시차와 사후 수정이 있습니다. 개인 공부용 참고 도구이며 투자 조언이 아니고 매매 타이밍을 알려주지 않습니다.
임계값·비중은 백테스트 전의 임시 설정값입니다. 개인·비상업 참고용으로만 이용하세요. 투자 판단과 책임은 본인에게 있습니다.
</footer>
</body></html>"""


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--tx:#1b1f24;--sub:#6b7280;--line:#e5e7eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--card:#181c22;--tx:#e8eaed;--sub:#9aa1ab;--line:#2a2f37}}
*{box-sizing:border-box}
body{margin:0 auto;background:var(--bg);color:var(--tx);font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;padding:20px;max-width:1200px}
h1{font-size:20px;margin:0 0 4px} .sub{color:var(--sub);font-size:12px;margin-bottom:16px}
h3{font-size:14px;margin:0 0 10px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 20px;margin-bottom:16px}
.hero .hrow{display:flex;flex-wrap:wrap;gap:36px;align-items:flex-start}
.lbl{color:var(--sub);font-size:12px;line-height:1.6}
.big{font-size:52px;font-weight:700;line-height:1.1}
.score{font-size:40px;font-weight:700;line-height:1.2} .score small{font-size:14px;color:var(--sub);font-weight:500}
.gauge{position:relative;margin:18px 0 6px;padding-top:16px}
.segs{display:flex;height:26px;border-radius:7px;overflow:hidden}
.segs div{flex:1;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:600}
.mark{position:absolute;top:-2px;transform:translateX(-50%);font-size:16px;line-height:1}
.ticks{display:flex;justify-content:space-between;font-size:11px;color:var(--sub);margin-top:3px}
.ticks em{font-style:normal}
.how{font-size:12px;color:var(--sub);margin-top:8px;line-height:1.6}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:860px){.two{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--sub);font-weight:500;font-size:12px}
tr.cur td{background:rgba(120,120,120,.14);font-weight:600}
.note{font-size:12px;color:var(--sub);margin-top:10px;line-height:1.6}
.bars{display:flex;gap:7px;align-items:flex-end;height:112px}
.bar{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;font-size:10px;color:var(--sub)}
.bar b{font-size:10px;color:var(--tx);font-weight:600;margin-bottom:2px}
.bar i{width:14px;border-radius:3px;display:block;margin-bottom:3px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
h2{font-size:13px;color:var(--sub);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.top{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}
.name{font-size:13px;font-weight:600}
.pill{color:#fff;font-size:11px;padding:2px 8px;border-radius:99px;white-space:nowrap}
.val{font-size:26px;font-weight:700;margin:6px 0 2px} .val small{font-size:11px;color:var(--sub);font-weight:400}
.chart{color:var(--tx)} .meta{font-size:11px;color:var(--sub);margin-top:4px;line-height:1.5}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin-bottom:6px}
.legend i{display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
details summary{cursor:pointer;font-weight:600;font-size:14px} details ul{margin:10px 0 0;padding-left:18px;font-size:13px;line-height:1.7}
footer{color:var(--sub);font-size:11px;margin-top:10px;line-height:1.7}
.fred{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px;background:var(--card)}
footer a{color:inherit}
"""


def main():
    raw = {k: fetch(v) for k, v in SERIES.items()}
    df = build_frame(raw)
    out = pathlib.Path(__file__).parent / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(render(df), encoding="utf-8")
    print("OK", df.index[-1].date(), "score", score_row(df.iloc[-1]))


if __name__ == "__main__":
    main()
