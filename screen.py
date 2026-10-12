"""Screen every downloaded company in one go:   python3 screen.py        (real filings)
                                                python3 screen.py sample
Prints L, N, D per company and year pair, plus a quality flag. A row is marked CHECK when the
extraction looks wrong (e.g. one year has far more paragraphs than the other, or few matches)."""
import json, sys
from pathlib import Path
import core

kind = sys.argv[1] if len(sys.argv) > 1 else "real"
root = Path(__file__).parent / "data" / kind
rows = []
for cdir in sorted((root / "filings").iterdir()):
    if not cdir.is_dir():
        continue
    years = sorted(p.stem for p in cdir.glob("*.htm"))
    fp = root / "facts" / f"{cdir.name}.json"
    facts = json.loads(fp.read_text()) if fp.exists() else None
    for y0, y1 in zip(years, years[1:]):
        try:
            r = core.analyze((cdir / f"{y0}.htm").read_bytes(), (cdir / f"{y1}.htm").read_bytes(),
                             facts, int(y0), int(y1), prefer_st=False)
        except Exception as e:
            print(f"{cdir.name:8s} {y0}->{y1}  ERROR: {str(e)[:90]}")
            continue
        m, o = r["meta"], r["overall"]
        ok = m["n_old"] > 0 and m["matched"] / m["n_old"] >= 0.5 and 0.6 <= m["n_new"] / m["n_old"] <= 1.6
        rows.append((o["D"], cdir.name, f"{y0}->{y1}", o["L"], o["N"], m, ok))

rows.sort(key=lambda x: -x[0])
print(f'{"company":8s} {"years":10s} {"L":>6s} {"N":>6s} {"D":>6s}  {"matched":>9s} {"del/add":>8s}  quality')
for D, name, yrs, L, N, m, ok in rows:
    n = "  n/a" if N is None else f"{N:+.2f}"
    print(f'{name:8s} {yrs:10s} {L:+.2f} {n:>6s} {D:6.3f}  {m["matched"]:4d}/{m["n_old"]:<4d} '
          f'{m["deleted"]:3d}/{m["added"]:<4d}  {"ok" if ok else "CHECK - extraction looks wrong, do not use"}')