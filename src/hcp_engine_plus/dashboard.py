"""Write a completely self-contained, static HTML dashboard (no web server)."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def write_dashboard(targets: pd.DataFrame, out: str | Path) -> Path:
    """Small client-side searchable report. Synthetic/real data are not transmitted."""
    cols = [c for c in ["hcp_id", "segment", "status", "region", "propensity",
                        "expected_value", "tier", "in_call_plan"] if c in targets.columns]
    if not cols:
        raise ValueError("No recognizable targeting columns")
    df = targets[cols].head(5000).copy()
    for col in ("propensity", "expected_value"):
        if col in df:
            df[col] = df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    th = "".join(f"<th>{escape(str(c).replace('_', ' ').title())}</th>" for c in cols)
    trs = []
    for row in df.itertuples(index=False, name=None):
        tds = "".join(f"<td>{escape(str(v))}</td>" for v in row)
        trs.append(f"<tr>{tds}</tr>")
    n_hcps = len(targets)
    n_plan = int(targets["in_call_plan"].fillna(False).astype(bool).sum()) if "in_call_plan" in targets else 0
    n_scored = int(targets["propensity"].notna().sum()) if "propensity" in targets else 0
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HCP Targeting | Local Report</title><style>
:root{{--ink:#172b45;--muted:#607087;--line:#dce4ec;--bg:#f5f7fa;--accent:#194a73}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);font:15px/1.5 system-ui,sans-serif;color:var(--ink)}}
main{{max-width:1220px;margin:32px auto;padding:0 24px}}h1{{font-size:30px;margin:0}}p{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}
.card,.panel{{background:white;border:1px solid var(--line);border-radius:10px;padding:20px}}
.card strong{{display:block;font-size:29px;color:var(--accent)}}.card small{{color:var(--muted)}}
input{{padding:12px;border:1px solid var(--line);border-radius:7px;width:100%;
max-width:430px;margin:8px 0 16px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{color:var(--muted);background:#f8fafc;position:sticky;top:0}}tr:hover{{background:#f5f9fc}}
footer{{margin-top:18px;color:var(--muted);font-size:12px}}
@media(max-width:620px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>HCP Targeting Report</h1>
<p>Local, read-only report • Do not share identifiable HCP data without authorization.</p>
<div class="cards"><div class="card"><strong>{n_hcps:,}</strong><small>HCP records</small></div>
<div class="card"><strong>{n_scored:,}</strong><small>Scored prospects</small></div>
<div class="card"><strong>{n_plan:,}</strong><small>Call-plan candidates</small></div></div>
<section class="panel"><h2>Targeting table</h2>
<p>Showing up to 5,000 rows. Search by ID, segment, status or region.</p>
<label for="search">Search records</label><br>
<input id="search" type="search" placeholder="Filter records...">
<div class="scroll"><table id="targets"><thead><tr>{th}</tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<footer>Scores rank likelihood, not the causal effect of an outreach action.
Review targeting for compliance.</footer>
</section></main><script>
const search = document.getElementById('search');
search.addEventListener('input', () => {{
 const query = search.value.toLowerCase();
 for (const row of document.querySelectorAll('#targets tbody tr')) {{
   row.hidden = !row.textContent.toLowerCase().includes(query);
 }}
}});
</script></body></html>'''
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
