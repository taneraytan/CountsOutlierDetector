"""Self-contained HTML audit report for a detector run.

Everything (chart included) is embedded in a single HTML string so the file
can be e-mailed or archived without external assets.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
       max-width: 1100px; color: #1c2b3a; }
h1 { color: #14304a; border-bottom: 3px solid #2d6cb0; padding-bottom: 0.4rem; }
h2 { color: #1f4068; margin-top: 2rem; }
table { border-collapse: collapse; font-size: 0.85rem; margin: 0.5rem 0; }
th, td { border: 1px solid #cfd8e3; padding: 0.3rem 0.6rem; text-align: left; }
th { background: #eef3f9; }
pre { background: #f5f7fa; padding: 1rem; border-radius: 6px; overflow-x: auto;
      font-size: 0.8rem; }
.meta { color: #5a6b7d; font-size: 0.9rem; }
img { max-width: 100%; }
"""


def _score_chart_png(scores: pd.Series) -> str:
    counts = scores.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(counts.index.astype(str), counts.values, color="#2d6cb0")
    ax.set_xlabel("Total score")
    ax.set_ylabel("Number of rows")
    ax.set_title("Score distribution")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def format_explanation_items(expl) -> list[str]:
    """Turn a stored explanation cell into human-readable lines.

    A cell is either ``""`` / ``[]`` or a sequence of ``[[columns], [values]]``
    pairs (possibly numpy arrays after a parquet round-trip).
    """
    lines: list[str] = []
    if isinstance(expl, str) or expl is None:
        return lines
    try:
        items = list(expl)
    except TypeError:
        return lines
    for item in items:
        try:
            cols, vals = list(item[0]), list(item[1])
        except (TypeError, IndexError, KeyError):
            continue
        if len(cols) != len(vals):
            continue
        lines.append(" AND ".join(f"{c} = {v}" for c, v in zip(cols, vals)))
    return lines


def build_html_report(results: dict, max_rows: int = 50) -> str:
    """Render a run-results dict (as stored in session state) to HTML."""
    label = html.escape(str(results.get("label") or "Analysis"))
    scores = results.get("scores")
    if scores is None:
        scores = pd.Series(dtype=int)
    scores = pd.Series(scores).fillna(0)
    flagged_count = int((scores > 0).sum())
    n_rows = len(scores)
    pct = flagged_count / max(n_rows, 1) * 100

    params = results.get("params") or {}
    params_df = pd.DataFrame(
        [{"Parameter": str(k), "Value": str(v)} for k, v in params.items()]
    )

    most = results.get("most_flagged")
    most_html = (
        most.head(max_rows).to_html(escape=True)
        if isinstance(most, pd.DataFrame) and not most.empty
        else "<p>No rows were flagged.</p>"
    )

    summary = results.get("summary")
    summary_html = (
        summary.to_html(escape=True, index=False)
        if isinstance(summary, pd.DataFrame) and not summary.empty
        else "<p>(no summary)</p>"
    )

    # Per-row explanations for the top flagged rows.
    expl_blocks: list[str] = []
    flagged_all = results.get("flagged_all")
    if (isinstance(most, pd.DataFrame) and not most.empty
            and isinstance(flagged_all, pd.DataFrame)):
        for row_idx in list(most.index[:max_rows]):
            if row_idx not in flagged_all.index:
                continue
            lines: list[str] = []
            for d in range(1, 7):
                col = f"{d}d Explanations"
                if col in flagged_all.columns:
                    for line in format_explanation_items(flagged_all.loc[row_idx, col]):
                        lines.append(f"{d}d: {html.escape(line)}")
            if lines:
                items = "".join(f"<li>{line}</li>" for line in lines)
                expl_blocks.append(f"<h3>Row {html.escape(str(row_idx))}</h3><ul>{items}</ul>")
    expl_html = "".join(expl_blocks) or "<p>No explanations available.</p>"

    chart_b64 = _score_chart_png(scores) if n_rows else ""
    chart_html = (
        f'<img src="data:image/png;base64,{chart_b64}" alt="Score distribution"/>'
        if chart_b64 else "<p>(no scores)</p>"
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_summary = html.escape(str(results.get("run_summary") or "(no run summary)"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Counts Outlier Detector report — {label}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Counts Outlier Detector — {label}</h1>
<p class="meta">Generated {generated} · {n_rows:,} rows analysed ·
{flagged_count:,} rows flagged ({pct:.2f}%)</p>

<h2>Parameters</h2>
{params_df.to_html(escape=True, index=False) if not params_df.empty else "<p>(none recorded)</p>"}

<h2>Score distribution</h2>
{chart_html}

<h2>Top flagged rows</h2>
{most_html}

<h2>Why these rows were flagged</h2>
{expl_html}

<h2>Detector summary</h2>
<pre>{run_summary}</pre>
{summary_html}
</body>
</html>"""
