# Narrative–Numbers Divergence Detector (midterm build)

Detects where a company's risk-factor **language softens** while its reported **numbers worsen**.

## Run it (10 minutes)
```bash
pip install -r requirements.txt
python make_sample.py            # synthetic test company (works offline)
streamlit run app.py             # opens the demo
```

## Run it on REAL filings (do this first today)
1. Open `sec_fetch.py`, edit `USER_AGENT` to your name + college email (SEC blocks anonymous requests).
2. `python sec_fetch.py AAPL MSFT TSLA F` (any tickers; downloads the last 3 10-Ks + XBRL facts each)
3. `streamlit run app.py` → choose **Real SEC filings** in the sidebar.
4. CLI check without the UI: `python core.py real AAPL 2023 2024`

> **Untested on live SEC data.** The pipeline was tested end-to-end on synthetic filings only
> (the build environment had no access to sec.gov). The most likely thing to need a tweak on real filings is
> Item 1A extraction (`_START` / `_END` regexes and `html_blocks()` in `core.py`). If a filing errors with
> "Could not locate Item 1A", print `core.html_blocks(open(file,'rb').read())[:200]` and look at how the heading appears.

## Files
| File | Job |
|---|---|
| `sec_fetch.py` | Download 10-Ks + XBRL company facts from SEC EDGAR |
| `core.py` | Extract Item 1A → align paragraphs across years → score softening (L) → XBRL numbers (N) → divergence (D) |
| `app.py` | Streamlit UI: divergence board, side-by-side evidence, numbers, method/status |
| `make_sample.py` | Generates a clearly-labelled SYNTHETIC test filing pair |

## Data layout
`data/<real|sample>/filings/<TICKER>/<FY>.htm` and `data/<real|sample>/facts/<TICKER>.json` (FY = fiscal period end year).

## What is and isn't built
Built: download, extraction, topic-constrained alignment, deletion/addition detection, linguistic score, XBRL numeric score, divergence, UI.
Not built (phase 2): ColPali, Neo4j, LangGraph agents + verifier, backtest, weight calibration. No predictive results are claimed.
