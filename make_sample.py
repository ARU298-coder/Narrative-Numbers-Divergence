"""Creates SYNTHETIC test filings (a fictional company) so the pipeline/app can run without
internet. This is NOT real data and must never be presented as a real company's filing."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "data" / "sample"
T = "ACME-SYNTH"

SUPPLY = "Risks Related to Our Supply Chain"
CUST = "Risks Related to Our Customers"
DEBT = "Risks Related to Our Indebtedness"
COMP = "Risks Related to Competition"
LEGAL = "Legal and Regulatory Risks"
IT = "Risks Related to Information Technology"

S1_23 = ("We depend on two suppliers, Northwind Semiconductor Corp. and Delta Precision Ltd., for approximately 41% of our component purchases, and the loss of either supplier will materially and adversely affect our ability to manufacture our products and will significantly reduce our revenue.")
S1_24 = ("We depend on certain third parties for a portion of our component purchases, and the loss of one of these suppliers may adversely affect our ability to manufacture our products and could reduce our revenue.")
S2_23 = ("Several of our key components are sourced from a single supplier with no qualified alternative. A disruption at Northwind Semiconductor Corp. would require us to redesign products, which will take at least nine months and will materially harm our results of operations.")
S2_24 = ("Some of our components may be sourced from a limited number of suppliers. A disruption could potentially require us to adjust our product designs, which may take time and could affect our results of operations in future periods.")
S3_23 = ("In fiscal 2023 we entered into a minimum purchase commitment of $64 million with Northwind Semiconductor Corp. that extends through 2026. If our demand falls below the committed volumes we will be required to pay shortfall penalties that will materially reduce our operating income.")
S4 = ("Our products are shipped from facilities in Penang, Malaysia and Monterrey, Mexico. Increases in freight costs, port congestion, or delays in customs clearance have increased our delivery times in the past and could continue to do so in the future.")
C1_23 = ("Our three largest customers accounted for 52% of our net revenue in fiscal 2023, and Meridian Automotive Group alone accounted for 27%. The loss of Meridian, or a significant reduction in its orders, will materially and adversely affect our revenue and profitability.")
C1_24 = ("A limited number of customers account for a substantial portion of our net revenue. A reduction in orders from one or more of these customers may adversely affect our revenue and our profitability in future periods.")
C2_23 = ("Demand for our products depends on the automotive and industrial end markets, which are cyclical. During the downturn of 2020 our revenue declined 18%, and we expect that a similar downturn will significantly reduce our revenue and cash flow.")
C2_24 = ("Demand for our products depends on the automotive and industrial end markets, which are cyclical. A downturn in these markets could potentially reduce our revenue and cash flow at some point in the future.")
L1_23 = ("Our credit agreement requires us to maintain a total leverage ratio below 3.5 times and contains covenants that restrict our ability to incur additional debt. A breach of these covenants will result in an event of default and will materially impair our liquidity.")
L1_24 = ("Our credit agreement contains financial covenants and restrictions that may limit our ability to incur additional debt. A breach of these covenants could result in an event of default and may affect our liquidity in some circumstances.")
L2_23 = ("We had $120 million of outstanding borrowings under our term loan as of the end of fiscal 2023. Our indebtedness requires us to dedicate a significant portion of our cash flow to debt service, which reduces funds available for capital expenditures.")
L2_24 = ("We may incur indebtedness from time to time. Debt service obligations could reduce the funds that we have available for capital expenditures, working capital, acquisitions and other general corporate purposes.")
K1_23 = ("We compete with larger companies that have greater financial resources. Competitive pressure may force us to reduce prices, which could reduce our gross margin and may adversely affect our results of operations in future periods.")
K1_24 = ("We compete with larger companies that have substantially greater financial resources. Competitive pressure will force us to reduce prices, which will significantly reduce our gross margin and will materially and adversely affect our results of operations.")
G1 = ("We are subject to export control and sanctions laws in the United States and other jurisdictions in which we operate. Violations of these laws could result in fines, loss of export privileges and damage to our reputation with customers and regulators.")
G2 = ("Changes in tax law, including changes to the treatment of foreign earnings, could increase our effective tax rate and reduce our net income. We monitor legislative developments in each jurisdiction in which we have operations.")
T1 = ("We rely on information technology systems to manage our manufacturing operations. A cybersecurity incident or data breach could disrupt production, expose confidential customer information and result in significant costs and legal liability.")
T2_24 = ("In fiscal 2024 we identified unauthorized access to a non-production system at our Monterrey facility. Although we do not believe the incident was material, similar incidents could occur and may disrupt our operations and expose us to liability.")

Y23 = [(SUPPLY, [S1_23, S2_23, S3_23, S4]), (CUST, [C1_23, C2_23]), (DEBT, [L1_23, L2_23]),
       (COMP, [K1_23]), (LEGAL, [G1, G2]), (IT, [T1])]
Y24 = [(SUPPLY, [S1_24, S2_24, S4]), (CUST, [C1_24, C2_24]), (DEBT, [L1_24, L2_24]),
       (COMP, [K1_24]), (LEGAL, [G1, G2]), (IT, [T1, T2_24])]


def doc(year, sections):
    h = [f"<html><body><div>ACME COMPONENTS INC. (SYNTHETIC TEST DATA) - FORM 10-K - FISCAL YEAR {year}</div>",
         "<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr>",
         "<tr><td>Item 1A. Risk Factors</td><td>9</td></tr>",
         "<tr><td>Item 1B. Unresolved Staff Comments</td><td>21</td></tr></table>",
         "<div>Item 1. Business</div>",
         "<div><span>We design and sell precision components. See </span><span>Item 1A. Risk Factors</span>"
         "<span> for a discussion of risks that may affect us.</span></div>",
         "<div><span>Item 1A.</span><span> Risk Factors</span></div>"]
    for head, paras in sections:
        h.append(f"<div><span><b>{head}</b></span></div>")
        h += [f"<div><span>{p[:60]}</span><span>{p[60:]}</span></div>" for p in paras]
    h += ["<div>Item 1B. Unresolved Staff Comments</div><div>None.</div>", "<div>Item 2. Properties</div>",
          "<div>Our headquarters are located in leased facilities.</div></body></html>"]
    return "".join(h)


def fact(tag, rows, duration=True):
    out = []
    for y, v in rows:
        f = {"end": f"{y}-12-31", "val": v, "form": "10-K", "fp": "FY", "filed": f"{y + 1}-02-20"}
        if duration:
            f["start"] = f"{y}-01-01"
        out.append(f)
    return tag, {"units": {"USD": out}}


if __name__ == "__main__":
    fd = OUT / "filings" / T
    fd.mkdir(parents=True, exist_ok=True)
    (OUT / "facts").mkdir(parents=True, exist_ok=True)
    (fd / "2023.htm").write_text(doc(2023, Y23))
    (fd / "2024.htm").write_text(doc(2024, Y24))
    g = dict([
        fact("Revenues", [(2023, 480e6), (2024, 452e6)]),
        fact("NetIncomeLoss", [(2023, 31e6), (2024, 18e6)]),
        fact("OperatingIncomeLoss", [(2023, 52e6), (2024, 38e6)]),
        fact("GrossProfit", [(2023, 150e6), (2024, 127e6)]),
        fact("LongTermDebt", [(2023, 120e6), (2024, 165e6)], False),
        fact("Assets", [(2023, 600e6), (2024, 640e6)], False),
        fact("CashAndCashEquivalentsAtCarryingValue", [(2023, 70e6), (2024, 44e6)], False)])
    (OUT / "facts" / f"{T}.json").write_text(json.dumps({"facts": {"us-gaap": g}}))
    print("sample written to", OUT)
