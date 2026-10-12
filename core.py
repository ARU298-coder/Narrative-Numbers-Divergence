"""Narrative-Numbers Divergence: midterm pipeline.
extract Item 1A -> align paragraphs across years -> score softening (L)
-> XBRL numeric deterioration (N) -> divergence D = max(0,-L) * max(0,N) * C
"""
import re, html, difflib
from datetime import date
import warnings
import numpy as np
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

clip = lambda x, lo=-1.0, hi=1.0: max(lo, min(hi, x))

# ------------------------------------------------------------------ 1. EXTRACTION
_START = re.compile(r"^\W*item\s*1a\W*\s*risk\s*factors", re.I)
_END = re.compile(r"^\W*item\s*(1b|1c|2)\b", re.I)


def html_blocks(raw):
    """Leaf-level text blocks (p/div/li/td) of a filing, whitespace-normalised.
    <br> tags are treated as block breaks."""
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\u0001")
    out = []
    for el in soup.find_all(["p", "div", "li", "td"]):
        if el.find(["p", "div", "li", "td", "table", "tr"]):
            continue
        for part in el.get_text(" ").split("\u0001"):
            t = re.sub(r"\s+", " ", part.replace("\xa0", " ")).strip()
            if t:
                out.append(t)
    return out


# Heading patterns. Real filings differ: "Item 1A. Risk Factors" in one block, split across
# table cells ("Item 1A." | "Risk Factors"), or just "Risk Factors" with no Item number.
_PLAIN_START = re.compile(r"^\W*risk\s*factors\W*$", re.I)
_PLAIN_END = re.compile(r"^\W*(unresolved\s+staff\s+comments|properties|legal\s+proceedings|mine\s+safety\s+disclosures|"
                        r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk|"
                        r"management.?s\s+discussion\s+and\s+analysis.*)\W*$", re.I)


def _heading_at(blocks, i, rx, maxk=3):
    """Number of consecutive short blocks (1..maxk) starting at i that together match rx, else 0."""
    parts = []
    for k in range(1, maxk + 1):
        if i + k > len(blocks) or len(blocks[i + k - 1]) > (120 if k == 1 else 60):
            return 0
        parts.append(blocks[i + k - 1])
        if rx.match(" ".join(parts)):
            return k
    return 0


def _best_span(blocks, srx, erx):
    ends = [i for i in range(len(blocks)) if _heading_at(blocks, i, erx, 1)]
    best = None
    for i in range(len(blocks)):
        k = _heading_at(blocks, i, srx)
        if k:
            e = next((x for x in ends if x >= i + k), None)
            if e is not None and (best is None or e - (i + k) > best[1] - best[0]):
                best = (i + k, e)
    return best


def risk_section(blocks):
    """Risk-factor section = LONGEST span from a start heading to the next end heading
    (longest span skips the table of contents / cross-reference index)."""
    for srx, erx in ((_START, _END), (_PLAIN_START, _PLAIN_END)):
        b = _best_span(blocks, srx, erx)
        if b and b[1] - b[0] >= 5:
            return blocks[b[0]:b[1]]
    return None


def diagnose(raw):
    """Print how headings look in a filing (use when extraction fails)."""
    blocks = html_blocks(raw)
    print(f"{len(blocks)} text blocks")
    for i, b in enumerate(blocks):
        if len(b) < 150 and re.search(r"risk\s*factors|^\W*item\s*(1a|1b|1c|2)\b|unresolved\s+staff", b, re.I):
            print(f"  [{i:5d}] len={len(b):3d}  {b!r}")


def paragraphs(blocks):
    """Body paragraphs (>=25 words); shorter blocks are treated as headings."""
    out, head = [], ""
    for b in blocks:
        if len(b.split()) < 25:
            head = b
            continue
        out.append({"text": b, "heading": head})
    return out


# ------------------------------------------------------------------ 2. TOPICS
TOPICS = {
    "Supply chain & suppliers": r"supplier|supply chain|sole.source|single.source|component|vendor|procure|logistic|freight",
    "Customers & demand": r"customer|demand|end market",
    "Competition": r"compet",
    "Liquidity, debt & capital": r"indebted|debt|credit (agreement|facility)|covenant|liquidity|borrow|financing|capital expend",
    "Regulation, legal & tax": r"regulat|legal|litigation|complian|export control|sanction|tax|privacy|government",
    "Cybersecurity & IT": r"cyber|data breach|information technology|ransomware|security incident",
    "Macro & operations": r"inflation|interest rate|economic|pandemic|geopolit|tariff|recession|currency|foreign exchange",
}
# which numeric group each topic is compared against (company-level XBRL data)
TOPIC_GROUP = {"Supply chain & suppliers": "margin", "Customers & demand": "revenue",
               "Competition": "margin_rev", "Liquidity, debt & capital": "leverage"}


def topic_of(p):
    best, score = "General", 0
    for name, rx in TOPICS.items():
        s = 2 * len(re.findall(rx, p["heading"], re.I)) + len(re.findall(rx, p["text"], re.I))
        if s > score:
            best, score = name, s
    return best


# ------------------------------------------------------------------ 3. ALIGNMENT
def embed(a, b, prefer_st=True):
    if prefer_st:
        try:
            from sentence_transformers import SentenceTransformer
            E = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
                a + b, normalize_embeddings=True, show_progress_bar=False)
            return E[:len(a)], E[len(a):], "sentence-transformers (MiniLM)", 0.55
        except Exception:
            pass
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(a + b)
    return v.transform(a), v.transform(b), "TF-IDF (fallback)", 0.30


def align(old, new, prefer_st=True):
    """Topic-constrained Hungarian matching on embedding similarity.
    - same-topic pairs are favoured and accepted at a LOWER similarity (a heavily softened
      rewrite is still the 'same' disclosure, not a deletion);
    - cross-topic pairs are accepted only if nearly identical (paragraph moved).
    Unmatched old = candidate deletions; unmatched new = newly added disclosures."""
    Ea, Eb, name, thr = embed([p["text"] for p in old], [p["text"] for p in new], prefer_st)
    S = Ea @ Eb.T
    S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    to, tn = [topic_of(p) for p in old], [topic_of(p) for p in new]
    same = np.array([[a == b for b in tn] for a in to])
    head = np.array([[0.05 if o["heading"] and o["heading"].lower() == n["heading"].lower() else 0.0
                      for n in new] for o in old])
    r, c = linear_sum_assignment(-(S + head + np.where(same, 0.15, -0.10)))
    thr_same, thr_cross = max(thr - 0.15, 0.10), thr + 0.20
    pairs = [(int(i), int(j), float(S[i, j])) for i, j in zip(r, c)
             if S[i, j] >= (thr_same if same[i, j] else thr_cross)]
    mo, mn = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    return (pairs, [i for i in range(len(old)) if i not in mo],
            [j for j in range(len(new)) if j not in mn], name, thr_same)


def diff_html(a, b):
    A, B = a.split(), b.split()
    ha, hb = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        x, y = html.escape(" ".join(A[i1:i2])), html.escape(" ".join(B[j1:j2]))
        if op == "equal":
            ha.append(x); hb.append(y)
        else:
            if x: ha.append(f'<span class="del">{x}</span>')
            if y: hb.append(f'<span class="ins">{y}</span>')
    return " ".join(ha), " ".join(hb)


# ------------------------------------------------------------------ 4. LINGUISTIC TRAJECTORY (L)
_HEDGE = re.compile(r"\b(may|might|could|potentially|possibly|generally|typically|certain|various|some|likely|perhaps|occasionally|from time to time|in some cases|to some extent)\b", re.I)
_QUAL = re.compile(r"\b(material|materially|significant|significantly|substantial|substantially|adverse|adversely|severe|severely|serious|seriously)\b", re.I)
_MODAL = [(re.compile(r"\b(will|shall|must)\b", re.I), 1.0),
          (re.compile(r"\b(expect|expects|anticipate|anticipates|intend|intends)\b", re.I), 0.66),
          (re.compile(r"\b(may|could|can)\b", re.I), 0.33),
          (re.compile(r"\b(might|possibly)\b", re.I), 0.2)]
_NUM = re.compile(r"\$?\d[\d,\.]*%?")
# initial weights (to be calibrated on a development set in phase 2)
W = dict(hedge=0.20, qual=0.15, modal=0.15, spec=0.25, delete=0.25)


def feats(text):
    toks = re.findall(r"\w[\w'\-]*", text)
    n = max(len(toks), 1)
    caps = sum(1 for s in re.split(r"(?<=[.!?])\s+", text) for w in s.split()[1:] if w[:1].isupper())
    mods = [s for rx, s in _MODAL for _ in rx.findall(text)]
    return dict(hedge=len(_HEDGE.findall(text)) * 100 / n, qual=len(_QUAL.findall(text)) * 100 / n,
                modal=(sum(mods) / len(mods)) if mods else None,
                spec=(caps + len(_NUM.findall(text))) * 100 / n)


def softening(a, b):
    """Per-pair softening in [-0.75, 0.75]; positive = language softened."""
    h = clip((b["hedge"] - a["hedge"]) / 5.0)
    q = clip((a["qual"] - b["qual"]) / 4.0)
    m = clip((a["modal"] - b["modal"]) / 0.5) if a["modal"] is not None and b["modal"] is not None else 0.0
    s = clip((a["spec"] - b["spec"]) / 10.0)
    return dict(hedging=h, qualifiers=q, modal=m, specificity=s,
                total=W["hedge"] * h + W["qual"] * q + W["modal"] * m + W["spec"] * s)


# ------------------------------------------------------------------ 5. NUMERIC TRAJECTORY (N) from XBRL
TAGS = {
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                 "RevenueFromContractWithCustomerIncludingAssessedTax"], True),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], True),
    "op_income": (["OperatingIncomeLoss"], True),
    "gross_profit": (["GrossProfit"], True),
    "debt": (["LongTermDebt", "LongTermDebtNoncurrent", "DebtInstrumentCarryingAmount"], False),
    "assets": (["Assets"], False),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], False),
}


def _annual(facts, tags, duration):
    out, gaap = {}, facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        best = {}
        for f in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if not str(f.get("form", "")).startswith("10-K"):
                continue
            if duration:
                if "start" not in f:
                    continue
                if not 350 <= (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days <= 380:
                    continue
            y = int(f["end"][:4])
            if y not in best or f["filed"] > best[y][1]:
                best[y] = (f["val"], f["filed"])
        for y, (v, _) in best.items():
            out.setdefault(y, v)
    return out


def numeric(facts, y0, y1):
    """Each metric -> deterioration in [-1,1] (positive = condition worsened)."""
    if not facts:
        return dict(rows=[], groups={})
    D = {k: _annual(facts, t, d) for k, (t, d) in TAGS.items()}
    g = lambda k, y: D[k].get(y)
    rows = []

    def add(name, o, n, det, groups, change, kind):
        rows.append(dict(metric=name, old=o, new=n, change=change, deterioration=det, groups=groups, kind=kind))

    r0, r1 = g("revenue", y0), g("revenue", y1)
    if r0 and r1:
        c = (r1 - r0) / abs(r0)
        add("Revenue", r0, r1, clip(-c / 0.20), ["revenue", "margin_rev", "overall"], f"{c:+.1%}", "usd")
    if r0 and r1:
        for key, label in (("gross_profit", "Gross margin"), ("op_income", "Operating margin")):
            a0, a1 = g(key, y0), g(key, y1)
            if a0 is not None and a1 is not None:
                m0, m1 = a0 / r0, a1 / r1
                add(label, m0, m1, clip(-(m1 - m0) / 0.05),
                    ["margin", "margin_rev", "overall"],
                    f"{(m1 - m0) * 100:+.1f} pp", "pct")
    n0, n1 = g("net_income", y0), g("net_income", y1)
    if n0 and n1 is not None:
        c = (n1 - n0) / abs(n0)
        add("Net income", n0, n1, clip(-c / 0.5), ["overall"], f"{c:+.1%}", "usd")
    a0, a1 = g("assets", y0), g("assets", y1)
    if a0 and a1:
        d0, d1, c0, c1 = g("debt", y0), g("debt", y1), g("cash", y0), g("cash", y1)
        if d0 is not None and d1 is not None:
            add("Debt / assets", d0 / a0, d1 / a1, clip(((d1 / a1) - (d0 / a0)) / 0.05),
                ["leverage", "overall"], f"{((d1 / a1) - (d0 / a0)) * 100:+.1f} pp", "pct")
        if c0 is not None and c1 is not None:
            add("Cash / assets", c0 / a0, c1 / a1, clip(-((c1 / a1) - (c0 / a0)) / 0.05),
                ["leverage", "overall"], f"{((c1 / a1) - (c0 / a0)) * 100:+.1f} pp", "pct")
    groups = {}
    for grp in ("revenue", "margin", "margin_rev", "leverage", "overall"):
        v = [r["deterioration"] for r in rows if grp in r["groups"]]
        if v:
            groups[grp] = float(np.mean(v))
    return dict(rows=rows, groups=groups)


# ------------------------------------------------------------------ 6. DIVERGENCE
def _score(t, num, group):
    tops = sorted((p["feat"]["total"] for p in t["pairs"]), reverse=True)[:3]
    pair_part = float(np.mean(tops)) if tops else 0.0
    d = clip(len(t["deleted"]) / t["n_old"] / 0.25, 0, 1) if t["n_old"] else 0.0
    L = -(pair_part + W["delete"] * d)                       # negative = language softened
    N = num["groups"].get(group, num["groups"].get("overall"))
    sims = [p["sim"] for p in t["pairs"]]
    C = (clip(float(np.mean(sims)) if sims else 0.5, 0, 1) *
         min(1.0, (len(t["pairs"]) + len(t["deleted"])) / 3)) if N is not None else 0.0
    t.update(L=L, N=N, C=C, D=max(0.0, -L) * max(0.0, N or 0.0) * C, deletion_term=d)
    return t


def analyze(old_raw, new_raw, facts, y0, y1, prefer_st=True):
    sec0, sec1 = risk_section(html_blocks(old_raw)), risk_section(html_blocks(new_raw))
    for sec, y in ((sec0, y0), (sec1, y1)):
        if not sec:
            raise ValueError(f"Could not locate the Risk Factors section in the FY{y} filing. "
                             f"Run:  python3 core.py diag <real|sample> <TICKER> {y}")
    po, pn = paragraphs(sec0), paragraphs(sec1)
    pairs, deleted, added, backend, thr = align(po, pn, prefer_st)
    num = numeric(facts, y0, y1)

    def new_t(name):
        return dict(topic=name, pairs=[], deleted=[], added=[], n_old=0)

    topics = {}
    for p in po:
        topics.setdefault(topic_of(p), new_t(topic_of(p)))["n_old"] += 1
    allt = new_t("ALL RISK FACTORS (overall)")
    allt["n_old"] = len(po)
    for i, j, s in pairs:
        t = topics[topic_of(po[i])]
        fa, fb = feats(po[i]["text"]), feats(pn[j]["text"])
        da, db = diff_html(po[i]["text"], pn[j]["text"])
        item = dict(sim=s, old=po[i], new=pn[j], diff_old=da, diff_new=db, fa=fa, fb=fb,
                    feat=softening(fa, fb), changed=po[i]["text"] != pn[j]["text"])
        t["pairs"].append(item); allt["pairs"].append(item)
    for i in deleted:
        topics[topic_of(po[i])]["deleted"].append(po[i]); allt["deleted"].append(po[i])
    for j in added:
        topics.setdefault(topic_of(pn[j]), new_t(topic_of(pn[j])))["added"].append(pn[j])
        allt["added"].append(pn[j])
    res = [_score(t, num, TOPIC_GROUP.get(t["topic"], "overall")) for t in topics.values()]
    res.sort(key=lambda t: t["D"], reverse=True)
    return dict(topics=res, overall=_score(allt, num, "overall"), numeric=num,
                meta=dict(n_old=len(po), n_new=len(pn), matched=len(pairs), deleted=len(deleted),
                          added=len(added), backend=backend, threshold=thr, y0=y0, y1=y1))


if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    if sys.argv[1] == "diag":
        _, _, kind, tk, y = sys.argv[:5]
        diagnose((Path(__file__).parent / "data" / kind / "filings" / tk / f"{y}.htm").read_bytes())
        sys.exit()
    kind, tk, y0, y1 = sys.argv[1:5]
    d = Path(__file__).parent / "data" / kind
    fp = d / "facts" / f"{tk}.json"
    r = analyze((d / "filings" / tk / f"{y0}.htm").read_bytes(), (d / "filings" / tk / f"{y1}.htm").read_bytes(),
                json.loads(fp.read_text()) if fp.exists() else None, int(y0), int(y1), prefer_st=False)
    print(r["meta"])
    for t in [r["overall"]] + r["topics"]:
        print(f'{t["topic"]:34s} L={t["L"]:+.2f} N={t["N"] if t["N"] is None else round(t["N"], 2)} '
              f'C={t["C"]:.2f} D={t["D"]:.3f}  pairs={len(t["pairs"])} del={len(t["deleted"])} add={len(t["added"])}')