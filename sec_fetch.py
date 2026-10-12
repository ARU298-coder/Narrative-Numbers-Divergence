"""Download recent 10-K filings + XBRL company facts from SEC EDGAR.
Usage:  python sec_fetch.py AAPL MSFT TSLA          (default: last 3 annual reports each)
        python sec_fetch.py AAPL --years 4

SEC requires a descriptive User-Agent with contact details and allows max 10 requests/second.
EDIT USER_AGENT BELOW before running.
"""
import argparse, json, time
from pathlib import Path
import httpx

USER_AGENT = "Shourya Capstone arunetu@gmail.com"      # <-- EDIT THIS
DATA = Path(__file__).parent / "data" / "real"


def get(url):
    time.sleep(0.2)                                              # stay well under 10 req/s
    r = httpx.get(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
                  timeout=90, follow_redirects=True)
    r.raise_for_status()
    return r


def cik_for(ticker, _cache={}):
    if not _cache:
        _cache.update(get("https://www.sec.gov/files/company_tickers.json").json())
    for v in _cache.values():
        if v["ticker"].upper() == ticker.upper():
            return int(v["cik_str"])
    raise SystemExit(f"Ticker {ticker} not found in SEC list")


def fetch(ticker, years=3):
    ticker, cik = ticker.upper(), cik_for(ticker)
    rec = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    out = DATA / "filings" / ticker
    out.mkdir(parents=True, exist_ok=True)
    (DATA / "facts").mkdir(parents=True, exist_ok=True)
    got = 0
    for form, acc, doc, rdate in zip(rec["form"], rec["accessionNumber"], rec["primaryDocument"], rec["reportDate"]):
        if form != "10-K":
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        (out / f"{rdate[:4]}.htm").write_bytes(get(url).content)   # file name = fiscal period END year
        print(f"  {ticker} FY{rdate[:4]}  <- {url}")
        got += 1
        if got >= years:
            break
    facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    (DATA / "facts" / f"{ticker}.json").write_text(json.dumps(facts))
    print(f"  {ticker}: XBRL facts saved ({got} filings)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--years", type=int, default=3)
    a = ap.parse_args()
    if "your.name@" in USER_AGENT:
        raise SystemExit("Edit USER_AGENT in sec_fetch.py first (SEC blocks anonymous requests).")
    for t in a.tickers:
        fetch(t, a.years)
