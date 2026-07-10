#!/usr/bin/env python3
"""Render the full evidence_battery.py JSONL into one presentation page.

Panels, in the order they answer the narrative:
  1. Noise floor       -- iterations & error for the easy task x3 (should be flat/near-zero)
  2. Repeat-to-convergence -- iterations & error for the hard task x5 (should trend down)
  3. Family generalization -- A/B/C, 6 variants each, iterations & error
  4. Control species    -- 3 reps (pre/mid/post) overlaid on the family panel's x-axis
                            position for visual comparison
  5. Interference check -- family A's v1 task, original vs repeat-after-B-and-C
  6. Escalation trigger  -- move id per rep, flagging any ".esc." (earned LLM move)

Usage:
  python3 scripts/plot_evidence_battery.py evidence_battery_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_learning_curve import svg_panel, SERIES_COLORS_LIGHT, CONTROL_LIGHT  # noqa: E402


def load(path):
    by_phase = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_phase[r["phase"]].append(r)
    for phase in by_phase:
        by_phase[phase].sort(key=lambda r: r["variant"])
    return by_phase


def single_series_panel(records, key, title, y_label, color=SERIES_COLORS_LIGHT[0]):
    fam = {"series": records}
    return svg_panel(fam, None, key, title, y_label, [color], CONTROL_LIGHT)


def escalation_table(records) -> str:
    rows = []
    for r in records:
        moves = ", ".join(m["move"] or "" for m in r["moves"]) or "(none)"
        flag = " ⟵ ESCALATED" if any(".esc." in (m["move"] or "") for m in r["moves"]) else ""
        rows.append(f"<tr><td>v{r['variant']}</td><td>{r['iterations']}</td>"
                    f"<td>{r['initial_worst_excess']}</td><td>{moves}{flag}</td></tr>")
    return ("<table class='esc-table'><thead><tr><th>rep</th><th>iterations</th>"
            "<th>initial excess</th><th>moves fired</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def interference_table(orig, repeat) -> str:
    def row(label, r):
        moves = ", ".join(f"{m['axis']}:{m['move']}" for m in r["moves"]) or "(none)"
        return (f"<tr><td>{label}</td><td>{r['iterations']}</td>"
                f"<td>{r['initial_worst_excess']}</td><td>{r['verdict']}</td><td>{moves}</td></tr>")
    return ("<table class='esc-table'><thead><tr><th>run</th><th>iterations</th>"
            "<th>initial excess</th><th>verdict</th><th>moves fired</th></tr></thead><tbody>"
            + row("original (v1, cold)", orig) + row("repeat (after B &amp; C ran)", repeat)
            + "</tbody></table>")


def build_html(by_phase) -> str:
    noise = single_series_panel(by_phase.get("noise_floor", []), "iterations",
                                "Noise floor: iterations (easy task x3)", "iterations")
    noise_err = single_series_panel(by_phase.get("noise_floor", []), "initial_worst_excess",
                                    "Noise floor: initial error (easy task x3)", "E excess")
    repeat = single_series_panel(by_phase.get("repeat_convergence", []), "iterations",
                                 "Repeat-to-convergence: iterations (hard task x5)", "iterations",
                                 color=SERIES_COLORS_LIGHT[2])
    repeat_err = single_series_panel(by_phase.get("repeat_convergence", []), "initial_worst_excess",
                                     "Repeat-to-convergence: initial error (hard task x5)",
                                     "E excess", color=SERIES_COLORS_LIGHT[2])

    fam_families = {
        "A_divergent_enumerate": by_phase.get("family_A", []),
        "B_sequential_pipeline": by_phase.get("family_B", []),
        "C_recursive_exhaustive": by_phase.get("family_C", []),
    }
    control_recs = by_phase.get("control", [])
    control_marker = control_recs[0] if control_recs else None
    fam_iter = svg_panel(fam_families, control_marker, "iterations",
                        "Family generalization: iterations (6 variants each)",
                        "iterations", SERIES_COLORS_LIGHT, CONTROL_LIGHT)
    fam_err = svg_panel(fam_families, control_marker, "initial_worst_excess",
                       "Family generalization: initial error (6 variants each)",
                       "E excess", SERIES_COLORS_LIGHT, CONTROL_LIGHT)

    control_rows = "".join(
        f"<tr><td>rep {i+1} ({['pre','mid','post'][i] if i < 3 else i+1})</td>"
        f"<td>{r['iterations']}</td><td>{r['initial_worst_excess']}</td>"
        f"<td>{r['verdict']}</td></tr>"
        for i, r in enumerate(control_recs)
    )
    control_table = ("<table class='esc-table'><thead><tr><th>when</th><th>iterations</th>"
                     "<th>initial excess</th><th>verdict</th></tr></thead><tbody>"
                     + control_rows + "</tbody></table>")

    interference_html = ""
    fam_a = by_phase.get("family_A", [])
    interference = by_phase.get("interference_check", [])
    if fam_a and interference:
        interference_html = interference_table(fam_a[0], interference[0])

    esc_html = escalation_table(by_phase.get("escalation_trigger", []))

    return f"""<title>Conductor-Delta: evidence battery</title>
<style>
  :root {{ --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
           --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
             --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); }}
  }}
  :root[data-theme="dark"] {{ --surface-1: #1a1a19; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --grid: #2c2c2a; --axis: #383835; }}
  :root[data-theme="light"] {{ --surface-1: #fcfcfb; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --grid: #e1e0d9; --axis: #c3c2b7; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          color: var(--text-primary); max-width: 1180px; margin: 32px auto; padding: 0 20px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 2px; }}
  h2 {{ font-size: 1.05rem; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 20px; }}
  p.sub {{ color: var(--text-secondary); max-width: 820px; }}
  .grid-wrap {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .panel {{ background: var(--surface-1); border-radius: 10px; }}
  .panel-title {{ font-size: 13px; fill: var(--text-primary); font-weight: 600; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .tick, .axis-label {{ font-size: 10px; fill: var(--muted); }}
  .series-line {{ fill: none; stroke-width: 2; }}
  .direct-label {{ font-size: 11px; font-weight: 600; }}
  .pt {{ cursor: pointer; }}
  #tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary);
              color: var(--surface-1); font-size: 12px; padding: 4px 8px; border-radius: 6px;
              opacity: 0; transition: opacity 0.1s; z-index: 10; }}
  table.esc-table {{ border-collapse: collapse; font-size: 13px; width: 100%; max-width: 900px; }}
  table.esc-table th, table.esc-table td {{ text-align: left; padding: 6px 12px;
              border-bottom: 1px solid var(--grid); }}
  table.esc-table th {{ color: var(--text-secondary); font-weight: 600; }}
</style>
<h1>Conductor-Delta: full evidence battery</h1>
<p class="sub">One session, six phases, each targeting a specific claim in the
learning narrative. Numbers reflect the tensor of the actually-delivered
answer (Pareto-best iteration), not a discarded intermediate attempt.</p>

<h2>0. Noise floor &mdash; easy task, nothing to fix, x3</h2>
<p class="sub">Calibrates every other panel: this is what pure LLM stochasticity
looks like when the descent loop has no work to do. Flat, near-zero lines here
mean any trend elsewhere is real signal, not measurement noise.</p>
<div class="grid-wrap">{noise}{noise_err}</div>

<h2>1. Repeat-to-convergence &mdash; identical hard task, x5</h2>
<p class="sub">Same wording every time removes the "different variant hit a
different axis" confound. A downward trend here is unambiguously the learned-
move store, since the input never changes.</p>
<div class="grid-wrap">{repeat}{repeat_err}</div>

<h2>3. Family generalization &mdash; 3 species, 6 surface-different variants each</h2>
<p class="sub">Surface wording differs on every run within a family; only the
fingerprint species repeats. Any within-family improvement here is genuine
structural transfer, not memorization &mdash; the literal text is never seen twice.</p>
<div class="grid-wrap">{fam_iter}{fam_err}</div>

<h2>2. Control species &mdash; never seen, run pre/mid/post</h2>
<p class="sub">If these three numbers stay roughly flat while family panels
above show a within-family trend, that rules out "the whole session just gets
easier" as the explanation.</p>
{control_table}

<h2>4. Interference check &mdash; family A's v1 task, repeated after B and C ran</h2>
<p class="sub">Confirms A's learned moves survive unrelated species traffic in
the shared store.</p>
{interference_html or "<p class='sub'>(no data for this phase)</p>"}

<h2>5. Escalation trigger &mdash; recursive-flow / shallow-depth gap, x4</h2>
<p class="sub">Targets a documented gap in the deterministic move table. A
move id containing <code>.esc.</code> means real LLM-authored escalation fired
&mdash; the most novel mechanism in the architecture, and the one with the
least evidence behind it before this battery.</p>
{esc_html}

<div id="tooltip"></div>
<script>
  const tip = document.getElementById('tooltip');
  document.querySelectorAll('.pt').forEach(el => {{
    el.addEventListener('mousemove', e => {{
      tip.textContent = el.getAttribute('data-tip');
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY + 12) + 'px';
      tip.style.opacity = 1;
    }});
    el.addEventListener('mouseleave', () => tip.style.opacity = 0);
  }});
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="evidence_battery_chart.html")
    args = ap.parse_args()

    by_phase = load(args.results)
    if not by_phase:
        print("no records found in", args.results)
        sys.exit(1)

    html = build_html(by_phase)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
