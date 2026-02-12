"""Evaluation metrics and HTML report generation for customer support LLM fine-tuning."""

import json
from pathlib import Path


# =============================================================================
# Evaluation & metrics
# =============================================================================

# Keywords that suggest a support-relevant response (for relevance score)
SUPPORT_KEYWORDS = (
    "order", "refund", "cancel", "ship", "help", "support", "return",
    "payment", "account", "track", "delivery", "policy", "assist",
)


def compute_evaluation_metrics(
    results: list, has_fine_tuned: bool
) -> dict:
    """Compute aggregate metrics for the evaluation report."""
    n = len(results)
    if n == 0:
        return {"num_prompts": 0}

    def avg_len(key: str) -> float:
        lengths = [len(r.get(key, "")) for r in results]
        return round(sum(lengths) / n, 1) if lengths else 0

    def relevance_score(key: str) -> float:
        hits = 0
        for r in results:
            text = (r.get(key) or "").lower()
            if any(kw in text for kw in SUPPORT_KEYWORDS):
                hits += 1
        return round(hits / n, 2) if n else 0

    metrics = {
        "num_prompts": n,
        "avg_length_base": avg_len("base_output") if has_fine_tuned else None,
        "avg_length_fine_tuned": avg_len("fine_tuned_output") if has_fine_tuned else None,
        "avg_length_ollama": avg_len("ollama_output") if not has_fine_tuned else None,
        "relevance_score_base": relevance_score("base_output") if has_fine_tuned else None,
        "relevance_score_fine_tuned": relevance_score("fine_tuned_output") if has_fine_tuned else None,
        "relevance_score_ollama": relevance_score("ollama_output") if not has_fine_tuned else None,
    }
    return metrics


def load_evaluation_payload(path: str | Path) -> tuple[list, dict]:
    """Load evaluation JSON; return (results list, metrics dict). Handles legacy list format."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        has_ft = bool(data and "fine_tuned_output" in (data[0] or {}))
        return data, compute_evaluation_metrics(data, has_fine_tuned=has_ft)
    results = data.get("results", [])
    metrics = data.get("metrics", {})
    if not metrics and results:
        has_ft = "fine_tuned_output" in (results[0] or {})
        metrics = compute_evaluation_metrics(results, has_fine_tuned=has_ft)
    return results, metrics


# =============================================================================
# HTML evaluation report
# =============================================================================


def generate_evaluation_report(
    evaluation_path: str | Path,
    output_html_path: str | Path,
    training_stats_path: str | Path | None = None,
    title: str = "Customer Support LLM — Evaluation Report",
):
    """Generate a self-contained HTML evaluation report for demos."""
    evaluation_path = Path(evaluation_path)
    output_html_path = Path(output_html_path)
    results, metrics = load_evaluation_payload(evaluation_path)
    training = {}
    if training_stats_path and Path(training_stats_path).exists():
        with open(training_stats_path, encoding="utf-8") as f:
            training = json.load(f)

    has_fine_tuned = bool(results and "fine_tuned_output" in (results[0] or {}))
    has_ollama = bool(results and "ollama_output" in (results[0] or {}))

    # Build loss curve data from log_history if present
    loss_steps: list[int] = []
    loss_values: list[float] = []
    for entry in training.get("log_history", []):
        if "loss" in entry:
            loss_steps.append(entry.get("step", len(loss_steps)))
            loss_values.append(round(entry["loss"], 4))

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def row_cells(r: dict) -> str:
        parts = [f'<td class="prompt-cell"><div>{esc(str(r.get("prompt", ""))[:200])}</div></td>']
        if has_fine_tuned:
            parts.append(f'<td><div>{esc(str(r.get("base_output", ""))[:300])}</div></td>')
            parts.append(f'<td><div>{esc(str(r.get("fine_tuned_output", ""))[:300])}</div></td>')
        if has_ollama:
            parts.append(f'<td><div>{esc(str(r.get("ollama_output", ""))[:300])}</div></td>')
        return "\n".join(parts)

    table_header = "<th>Prompt</th>"
    if has_fine_tuned:
        table_header += "<th>Base model</th><th>Fine-tuned</th>"
    if has_ollama:
        table_header += "<th>Ollama</th>"

    rel_base = metrics.get("relevance_score_base") or metrics.get("relevance_score_ollama")
    rel_ft = metrics.get("relevance_score_fine_tuned")
    avg_base = metrics.get("avg_length_base") or metrics.get("avg_length_ollama")
    avg_ft = metrics.get("avg_length_fine_tuned")
    num_prompts = metrics.get("num_prompts", len(results))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0f0f12;
      --surface: #18181c;
      --surface2: #222228;
      --text: #e4e4e7;
      --textMuted: #a1a1aa;
      --accent: #22d3ee;
      --accentDim: #0891b2;
      --success: #34d399;
      --border: #27272a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'DM Sans', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 0;
      line-height: 1.6;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; margin: 0 0 0.5rem; letter-spacing: -0.02em; }}
    h2 {{ font-size: 1.25rem; font-weight: 600; margin: 2rem 0 1rem; color: var(--accent); }}
    .hero {{
      background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2rem;
      margin-bottom: 2rem;
    }}
    .hero p {{ margin: 0; color: var(--textMuted); font-size: 0.95rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 1rem;
      margin: 1rem 0;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
    }}
    .card h3 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--textMuted); margin: 0 0 0.5rem; }}
    .card .value {{ font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--accent); }}
    .card .value.good {{ color: var(--success); }}
    .section {{ margin-bottom: 2rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      background: var(--surface);
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background: var(--surface2); color: var(--textMuted); font-weight: 600; }}
    td div {{ max-height: 120px; overflow-y: auto; }}
    .prompt-cell div {{ max-width: 280px; }}
    .foot-note {{ font-size: 0.85rem; color: var(--textMuted); margin-top: 1rem; }}
    .loss-chart {{ height: 200px; background: var(--surface2); border-radius: 10px; padding: 1rem; border: 1px solid var(--border); }}
    .loss-chart svg {{ width: 100%; height: 100%; }}
    code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85em; background: var(--surface2); padding: 0.2em 0.4em; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>{esc(title)}</h1>
      <p>Fine-tuning evaluation and model statistics — before/after comparison and LoRA/QLoRA metrics.</p>
    </div>

    <section class="section">
      <h2>Training statistics (LoRA / QLoRA)</h2>
"""
    if training:
        trainable = training.get("trainable_params") or 0
        total = training.get("total_params") or 0
        pct = training.get("trainable_pct") or (round(100 * trainable / total, 2) if total else 0)
        duration = training.get("duration_seconds")
        duration_str = f"{duration:.1f}s" if isinstance(duration, (int, float)) else str(duration) if duration else "—"
        html += f"""
      <div class="grid">
        <div class="card"><h3>Trainable params</h3><div class="value">{trainable:,}</div></div>
        <div class="card"><h3>Total params</h3><div class="value">{total:,}</div></div>
        <div class="card"><h3>Trainable %</h3><div class="value">{pct}%</div></div>
        <div class="card"><h3>Duration</h3><div class="value">{duration_str}</div></div>
      </div>
"""
        cfg = training.get("config", {})
        if cfg:
            html += '      <div class="card" style="margin-top:1rem"><h3>Config</h3><pre style="margin:0;font-size:0.85rem;color:var(--textMuted);">'
            html += esc(json.dumps(cfg, indent=2))
            html += "</pre></div>\n"
        if loss_steps and loss_values:
            # Simple SVG line chart
            w, h = 800, 180
            pad = 40
            xs = [pad + (w - 2 * pad) * (i / max(len(loss_steps) - 1, 1)) for i in range(len(loss_steps))]
            mn, mx = min(loss_values), max(loss_values) or 1
            ys = [h - pad - (h - 2 * pad) * (v - mn) / (mx - mn or 1) for v in loss_values]
            pts = " ".join(f"{x},{y}" for x, y in zip(xs, ys))
            html += f'      <div class="loss-chart"><svg viewBox="0 0 {w} {h}"><polyline fill="none" stroke="var(--accent)" stroke-width="2" points="{pts}"/></svg></div>\n'
    else:
        html += '      <p class="foot-note">No training stats file provided. Run <code>train</code> and pass <code>--training_stats</code> to include LoRA/QLoRA stats.</p>\n'

    html += """
    </section>

    <section class="section">
      <h2>Evaluation metrics</h2>
      <div class="grid">
"""
    if rel_base is not None:
        html += f'        <div class="card"><h3>Relevance (base/Ollama)</h3><div class="value good">{rel_base}</div><span class="foot-note">support-keyword hit rate</span></div>\n'
    if rel_ft is not None:
        html += f'        <div class="card"><h3>Relevance (fine-tuned)</h3><div class="value good">{rel_ft}</div><span class="foot-note">support-keyword hit rate</span></div>\n'
    if avg_base is not None:
        html += f'        <div class="card"><h3>Avg length (base/Ollama)</h3><div class="value">{avg_base}</div><span class="foot-note">chars</span></div>\n'
    if avg_ft is not None:
        html += f'        <div class="card"><h3>Avg length (fine-tuned)</h3><div class="value">{avg_ft}</div><span class="foot-note">chars</span></div>\n'
    html += f'        <div class="card"><h3>Prompts evaluated</h3><div class="value">{num_prompts}</div></div>\n'
    html += """      </div>
      <p class="foot-note">Relevance: fraction of responses containing support-related keywords. Consistency: same prompt → similar behavior (qualitative).</p>
    </section>

    <section class="section">
      <h2>Scalability & maintainability</h2>
      <div class="grid">
        <div class="card"><h3>Scalability</h3><p style="margin:0;font-size:0.9rem;">QLoRA keeps memory low by training only adapters; same pipeline scales to larger base models or more data.</p></div>
        <div class="card"><h3>Maintainability</h3><p style="margin:0;font-size:0.9rem;">Adapter-only checkpoint is small and versioned; base model stays fixed for easy updates and A/B tests.</p></div>
      </div>
    </section>

    <section class="section">
      <h2>Before / after comparison</h2>
      <table>
        <thead><tr>
"""
    html += "          " + table_header + "\n        </tr></thead>\n        <tbody>\n"
    for r in results[:15]:
        html += "          <tr>\n            " + row_cells(r).replace("\n", "\n            ") + "\n          </tr>\n"
    html += "        </tbody>\n      </table>\n"
    if len(results) > 15:
        html += f'      <p class="foot-note">Showing first 15 of {len(results)} prompts. Full data in <code>{esc(str(evaluation_path))}</code>.</p>\n'
    html += """
    </section>
  </div>
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to {output_html_path}")
