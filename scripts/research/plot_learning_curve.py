#!/usr/bin/env python3
"""Turn cross_run_learning_test.py's JSONL into a presentation chart.

Two small-multiple line panels (never dual-axis):
  - iterations-to-converge, per family, across variant order (1..3)
  - initial worst-axis excess, per family, across variant order (1..3)

The control task (never-before-seen species, run last) is plotted as an
isolated reference point at x=1 on both panels in a neutral muted color --
if it lands near where variant-1 points of the real families landed, that's
the confound check: novelty is expensive regardless of session position, so
the real families' improvement across variants 1->3 is species-specific.

Usage:
  python3 scripts/plot_learning_curve.py cross_run_learning_results.jsonl
  python3 scripts/plot_learning_curve.py cross_run_learning_results.jsonl --out chart.html
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

# Fixed categorical order (dataviz skill palette.md) -- never reassigned per-render.
SERIES_COLORS_LIGHT = ["#2a78d6", "#1baf7a", "#eda100"]
SERIES_COLORS_DARK = ["#3987e5", "#199e70", "#c98500"]
CONTROL_LIGHT = "#898781"
CONTROL_DARK = "#898781"


def load(path):
    families = defaultdict(list)
    control = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["family"].startswith("D_control"):
                control = r
            else:
                families[r["family"]].append(r)
    for fam in families:
        families[fam].sort(key=lambda r: r["variant"])
    return families, control


def svg_panel(families, control, key, title, y_label, colors, control_color,
              width=560, height=280, pad=56):
    fam_names = sorted(families.keys())
    all_vals = [r[key] for fam in fam_names for r in families[fam]]
    if control:
        all_vals.append(control[key])
    y_max = max(all_vals) * 1.15 if all_vals else 1.0
    y_max = max(y_max, 1e-6)
    n_x = max((len(families[f]) for f in fam_names), default=3)
    n_x = max(n_x, 3)

    def xy(i, v):
        x = pad + (i - 1) / max(n_x - 1, 1) * (width - pad - 20)
        y = height - pad - (v / y_max) * (height - pad - 20)
        return x, y

    parts = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" class="panel" role="img" '
                 f'aria-label="{title}">')
    parts.append(f'<text x="{pad}" y="24" class="panel-title">{title}</text>')

    # gridlines + y-axis ticks
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gy = height - pad - frac * (height - pad - 20)
        gv = frac * y_max
        parts.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{width-20}" y2="{gy:.1f}" '
                     f'class="grid" />')
        parts.append(f'<text x="{pad-8}" y="{gy+4:.1f}" class="tick" text-anchor="end">'
                     f'{gv:.2f}</text>')

    # x-axis labels (variant order)
    for i in range(1, n_x + 1):
        x, _ = xy(i, 0)
        parts.append(f'<text x="{x:.1f}" y="{height-pad+20}" class="tick" '
                     f'text-anchor="middle">v{i}</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" class="axis-label" '
                 f'text-anchor="middle">variant order (surface-different wording, same species)</text>')
    parts.append(f'<text x="16" y="{height/2}" class="axis-label" '
                 f'text-anchor="middle" transform="rotate(-90 16 {height/2})">{y_label}</text>')

    # series lines + points
    for si, fam in enumerate(fam_names):
        color = colors[si % len(colors)]
        pts = families[fam]
        path = " ".join(
            f'{"M" if i == 0 else "L"}{xy(r["variant"], r[key])[0]:.1f},{xy(r["variant"], r[key])[1]:.1f}'
            for i, r in enumerate(pts)
        )
        parts.append(f'<path d="{path}" class="series-line" style="stroke:{color}" />')
        for r in pts:
            x, y = xy(r["variant"], r[key])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" style="fill:{color}" '
                f'class="pt" data-tip="{fam} v{r["variant"]}: {y_label}={r[key]:.3f}" />'
            )
        # direct label at last point (<=4 series total incl. control -> allowed)
        lx, ly = xy(pts[-1]["variant"], pts[-1][key])
        parts.append(f'<text x="{lx+8:.1f}" y="{ly+4:.1f}" class="direct-label" '
                     f'style="fill:{color}">{fam.split("_",1)[-1]}</text>')

    if control:
        x, y = xy(1, control[key])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" style="fill:none;'
                     f'stroke:{control_color};stroke-width:2.5" '
                     f'class="pt" data-tip="control (never seen): {y_label}={control[key]:.3f}" />')
        parts.append(f'<text x="{x+10:.1f}" y="{y-8:.1f}" class="direct-label" '
                     f'style="fill:{control_color}">control (cold)</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def build_html(families, control) -> str:
    panel1 = svg_panel(families, control, "iterations", "Iterations to converge",
                       "iterations", SERIES_COLORS_LIGHT, CONTROL_LIGHT)
    panel2 = svg_panel(families, control, "initial_worst_excess",
                       "First-run worst-axis error (before any repair)",
                       "E excess over threshold", SERIES_COLORS_LIGHT, CONTROL_LIGHT)

    return f"""<title>Conductor-Delta: cross-run learning &amp; generalization</title>
<style>
  :root {{
    --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
             --muted: #898781; --grid: #2c2c2a; --axis: #383835; }}
  }}
  :root[data-theme="dark"] {{ --surface-1: #1a1a19; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --grid: #2c2c2a; --axis: #383835; }}
  :root[data-theme="light"] {{ --surface-1: #fcfcfb; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --grid: #e1e0d9; --axis: #c3c2b7; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          color: var(--text-primary); max-width: 1180px; margin: 32px auto; padding: 0 20px; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 4px; }}
  p.sub {{ color: var(--text-secondary); margin-top: 0; max-width: 760px; }}
  .grid-wrap {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .panel {{ background: var(--surface-1); border-radius: 10px; overflow: visible; }}
  .panel-title {{ font-size: 13px; fill: var(--text-primary); font-weight: 600; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .tick {{ font-size: 10px; fill: var(--muted); }}
  .axis-label {{ font-size: 10px; fill: var(--muted); }}
  .series-line {{ fill: none; stroke-width: 2; }}
  .direct-label {{ font-size: 11px; font-weight: 600; }}
  .pt {{ cursor: pointer; }}
  #tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary);
              color: var(--surface-1); font-size: 12px; padding: 4px 8px; border-radius: 6px;
              opacity: 0; transition: opacity 0.1s; z-index: 10; }}
  .note {{ font-size: 12px; color: var(--text-secondary); max-width: 900px; margin-top: 20px; }}
</style>
<h1>Cross-run learning &amp; task-family generalization</h1>
<p class="sub">Each family runs 3 surface-different tasks that share one fingerprint
species. Improvement across v1&rarr;v3 cannot be memorization &mdash; the system
never sees the literal wording twice &mdash; so it is direct evidence of structural
transfer via the learned-move store, keyed on fingerprint shape alone.</p>
<div class="grid-wrap">
{panel1}
{panel2}
</div>
<p class="note">Control point (v1, muted ring) is a never-before-seen species run
last in the session. If it lands near where the real families' own v1 points sit,
that rules out "the session just gets easier over time" as the explanation for the
v1&rarr;v3 improvement within each family.</p>
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
    ap.add_argument("results", help="path to cross_run_learning_results.jsonl")
    ap.add_argument("--out", default="cross_run_learning_chart.html")
    args = ap.parse_args()

    families, control = load(args.results)
    if not families:
        print("no family records found in", args.results)
        sys.exit(1)

    html = build_html(families, control)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
