import json
from pathlib import Path
import pandas as pd
import streamlit as st
import core

ROOT = Path(__file__).parent / "data"
st.set_page_config(page_title="Narrative–Numbers Divergence", layout="wide")
st.markdown("""<style>
.del{background:#f8cfcf;text-decoration:line-through;padding:0 2px;border-radius:2px}
.ins{background:#c8ecd8;padding:0 2px;border-radius:2px}
.box{border:1px solid #d9dee5;border-radius:8px;padding:12px 14px;font-size:.92rem;line-height:1.6;color:#1a1a1a;background:#fff}
.gone{border-left:5px solid #c0392b;background:#fdf2f1}.new{border-left:5px solid #1e8449;background:#f1faf5}
</style>""", unsafe_allow_html=True)

st.title("Narrative–Numbers Divergence Detector")
st.caption("Where does a company's risk-disclosure language soften while its reported numbers worsen?")

# ---------------------------------------------------------------- sidebar
kinds = [k for k in ("real", "sample") if (ROOT / k / "filings").exists()]
if not kinds:
    st.error("No data found. Run `python make_sample.py` or `python sec_fetch.py AAPL`."); st.stop()
kind = st.sidebar.radio("Data source", kinds, format_func=lambda k: {
    "real": "Real SEC filings", "sample": "Synthetic sample (software test only)"}[k])
tickers = sorted(p.name for p in (ROOT / kind / "filings").iterdir() if p.is_dir())
tk = st.sidebar.selectbox("Company", tickers)
years = sorted(p.stem for p in (ROOT / kind / "filings" / tk).glob("*.htm"))
if len(years) < 2:
    st.error("Need at least two fiscal years of filings for this company."); st.stop()
y0 = st.sidebar.selectbox("Earlier year", years[:-1], index=len(years) - 2)
y1 = st.sidebar.selectbox("Later year", [y for y in years if y > y0])
use_st = st.sidebar.checkbox("Use sentence-transformers if installed", value=True,
                             help="Better paragraph matching. Falls back to TF-IDF automatically.")

if kind == "sample":
    st.warning("**SYNTHETIC SAMPLE DATA** — a fictional company generated to test the software. "
               "Not a real filing; do not present results from it as findings.")


@st.cache_data(show_spinner="Running pipeline…")
def run(kind, tk, y0, y1, use_st):
    d = ROOT / kind
    fp = d / "facts" / f"{tk}.json"
    return core.analyze((d / "filings" / tk / f"{y0}.htm").read_bytes(),
                        (d / "filings" / tk / f"{y1}.htm").read_bytes(),
                        json.loads(fp.read_text()) if fp.exists() else None, int(y0), int(y1), use_st)


try:
    R = run(kind, tk, y0, y1, use_st)
except ValueError as e:
    st.error(str(e)); st.stop()
M, O, NUM = R["meta"], R["overall"], R["numeric"]

# ---------------------------------------------------------------- headline
c = st.columns(5)
c[0].metric("Language trajectory L", f"{O['L']:+.2f}", help="Negative = disclosure language softened")
c[1].metric("Numeric trajectory N", "n/a" if O["N"] is None else f"{O['N']:+.2f}",
            help="Positive = reported numbers worsened (from SEC XBRL)")
c[2].metric("Divergence D", f"{O['D']:.3f}", help="max(0,-L) × max(0,N) × confidence")
c[3].metric("Paragraphs matched", f"{M['matched']} / {M['n_old']}")
c[4].metric("Deleted / Added", f"{M['deleted']} / {M['added']}")
st.caption(f"{tk}: FY{y0} → FY{y1} · alignment: {M['backend']} · min similarity (same topic) {M['threshold']:.2f}")

t1, t2, t3, t4 = st.tabs(["① Divergence board", "② Evidence (side by side)", "③ Numbers (SEC XBRL)", "④ Method & status"])

# ---------------------------------------------------------------- tab 1
with t1:
    df = pd.DataFrame([dict(Topic=t["topic"], L=round(t["L"], 2),
                            N=None if t["N"] is None else round(t["N"], 2), Confidence=round(t["C"], 2),
                            Divergence=round(t["D"], 3), Matched=len(t["pairs"]),
                            Deleted=len(t["deleted"]), Added=len(t["added"])) for t in R["topics"]])
    st.dataframe(df, hide_index=True)
    st.markdown("**Reading the board:** a topic scores only when its language *softened* (L < 0) **and** the linked "
                "numbers *worsened* (N > 0). Language that got stronger, or numbers that improved, score zero.")
    flagged = [t for t in R["topics"] if t["D"] > 0]
    if flagged:
        t = flagged[0]
        st.info(f"**Top divergence: {t['topic']}** — L = {t['L']:+.2f}, N = {t['N']:+.2f}, D = {t['D']:.3f}. "
                f"{len(t['deleted'])} deleted, {len(t['added'])} added disclosure(s). See tab ②.")

# ---------------------------------------------------------------- tab 2
with t2:
    names = [t["topic"] for t in R["topics"]]
    sel = st.selectbox("Topic", names)
    T = next(t for t in R["topics"] if t["topic"] == sel)
    changed = sorted([p for p in T["pairs"] if p["changed"]], key=lambda p: -p["feat"]["total"])
    st.markdown(f"**{len(changed)} changed paragraph(s)**, {len(T['pairs']) - len(changed)} unchanged, "
                f"**{len(T['deleted'])} deleted**, **{len(T['added'])} added.**")
    for p in changed:
        f, a, b = p["feat"], p["fa"], p["fb"]
        with st.expander(f"Softening {f['total']:+.2f}  ·  similarity {p['sim']:.2f}  ·  {p['old']['text'][:70]}…",
                         expanded=p is changed[0]):
            l, r = st.columns(2)
            l.markdown(f"**FY{y0}**<div class='box'>{p['diff_old']}</div>", unsafe_allow_html=True)
            r.markdown(f"**FY{y1}**<div class='box'>{p['diff_new']}</div>", unsafe_allow_html=True)
            mod = lambda x: "–" if x is None else f"{x:.2f}"
            st.caption(f"hedging {a['hedge']:.1f}→{b['hedge']:.1f}/100w  ·  qualifiers {a['qual']:.1f}→{b['qual']:.1f}/100w  ·  "
                       f"modal strength {mod(a['modal'])}→{mod(b['modal'])}  ·  specificity {a['spec']:.1f}→{b['spec']:.1f}/100w")
    for p in T["deleted"]:
        st.markdown(f"<div class='box gone'><b>⚠ In FY{y0}, no counterpart found in FY{y1} (candidate silent deletion):</b><br>"
                    f"{p['text']}</div><br>", unsafe_allow_html=True)
    for p in T["added"]:
        st.markdown(f"<div class='box new'><b>＋ New in FY{y1}:</b><br>{p['text']}</div><br>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 3
with t3:
    if not NUM["rows"]:
        st.info("No XBRL facts file for this company (data/<kind>/facts/<TICKER>.json).")
    else:
        def fmt(v, kind): return f"{v:.1%}" if kind == "pct" else f"${v / 1e6:,.1f}M"
        st.dataframe(pd.DataFrame([dict(Metric=r["metric"], **{f"FY{y0}": fmt(r["old"], r["kind"]),
                                                              f"FY{y1}": fmt(r["new"], r["kind"])},
                                        Change=r["change"], Deterioration=round(r["deterioration"], 2))
                                   for r in NUM["rows"]]), hide_index=True)
        st.bar_chart(pd.DataFrame({"Deterioration (+ = worse)": {r["metric"]: r["deterioration"] for r in NUM["rows"]}}))
        st.caption("Source: SEC XBRL company facts (non-dimensional annual 10-K values). "
                   "Each metric is scaled to [-1, +1]; topic groups average the relevant metrics.")

# ---------------------------------------------------------------- tab 4
with t4:
    st.markdown(r"""
**Formula** &nbsp; `D = max(0, −L) × max(0, N) × C`  
`L` = language trajectory (negative = softened): hedging, qualifier removal, modal-strength drop, specificity loss, silent deletion.  
`N` = numeric trajectory (positive = worsened) from SEC XBRL.  `C` = confidence (alignment quality × evidence size).

| Component | Status |
|---|---|
| SEC EDGAR download (10-K + XBRL facts) | ✅ built |
| Item 1A extraction and paragraph segmentation | ✅ built |
| Cross-year alignment (topic-constrained Hungarian matching) → deletion / addition detection | ✅ built |
| Linguistic softening score L (5 features) | ✅ built (weights are initial defaults) |
| Numeric trajectory N from XBRL | ✅ built (company-level metrics) |
| Divergence score + side-by-side evidence UI | ✅ built |
| ColPali visual retrieval (numbers inside charts) | ⏳ phase 2 |
| Neo4j temporal knowledge graph | ⏳ phase 2 |
| LangGraph agents + numeric verifier / abstention | ⏳ phase 2 |
| Forward backtest vs. later adverse events (pre-registered) | ⏳ phase 2 |
| Weight calibration on a development set | ⏳ phase 2 |

**Known limitations (midterm version):** numeric side is company-level, not topic-specific; weights are not yet calibrated;
topic tagging is keyword-based; no results on prediction are claimed. A divergence event indicates a *discrepancy between
language and reported figures* — it does **not** indicate misconduct.
""")
