"""Render the planner-comparison artifact page to PNGs and dump its chart data.

Input: the saved index.html (charts are drawn client-side by the page's own JS).
Output dir gets: full page (bf=4 and bf=64), one PNG per section, one PNG per chart,
and chart-data.json / chart-data-bf4.csv / chart-data-bf64.csv taken from the page's
strategies() model by patching a data dump into the page before rendering.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
WORK = OUT / "_work"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(exist_ok=True)
html = SRC.read_text(encoding="utf-8").replace("<html>", '<html data-theme="light">', 1)
assert 'data-theme="light"' in html
# Figure-export layout only (data untouched): a wider label gutter so the longest row label is
# not clipped, and a reserved right gutter so the hardware column never overlaps a bar's value.
html = html.replace(
    "const RW=560,L=168,R=16,T=10,B=34,rh=22", "const RW=640,L=240,R=110,T=10,B=34,rh=22", 1
)
html = html.replace(
    'el("text",{x:RW-R,y:cy+4,"text-anchor":"end"},svg);t.setAttribute("fill","var(--muted)");t.textContent=it.hw',
    'el("text",{x:RW-4,y:cy+4,"text-anchor":"end"},svg);t.setAttribute("fill","var(--muted)");t.textContent=it.hw',
    1,
)
assert "RW=640" in html and "x:RW-4" in html, "hbar layout anchors not found"

# Sections of <main>, in order, with file names.
SECTIONS = [
    ("00-header", "header"),
    ("01-leaderboard-and-efficiency", "section:nth-of-type(1)"),
    ("02-mechanisms-beyond-minimax", "section:nth-of-type(2)"),
    ("03-winner-per-metric", "section:nth-of-type(3)"),
    ("04-the-stack", "section:nth-of-type(4)"),
    ("05-genix-mind-rows", "section:nth-of-type(5)"),
    ("06-five-metrics", "section:nth-of-type(6)"),
    ("07-hybrid-evaluators", "section:nth-of-type(7)"),
    ("08-mechanism-by-mechanism", "section:nth-of-type(8)"),
]
CHARTS = ["chartLB", "chartEff", "chartMult", "chartLat", "chartPly", "chartNeed", "chartActs"]
CHART_NAMES = {
    "chartLB": "leaderboard-public-score",
    "chartEff": "per-level-score-vs-action-multiple",
    "chartMult": "metric1-action-multiple",
    "chartLat": "metric2-seconds-per-action",
    "chartPly": "metric3-simulated-plies",
    "chartNeed": "metric4-compute-minutes-per-game",
    "chartActs": "metric5-actions-affordable",
}


def with_bf(page: str, bf: int) -> str:
    if bf == 4:
        return page
    page = page.replace("let bf=4;", f"let bf={bf};")
    page = page.replace('data-b="4" aria-pressed="true"', 'data-b="4" aria-pressed="false"')
    return page.replace('data-b="64" aria-pressed="false"', 'data-b="64" aria-pressed="true"')


def only(page: str, css: str) -> str:
    """Hide every top-level block of <main> except the one matching ``css``."""
    style = f"<style>main>*{{display:none!important}}main>{css}{{display:grid!important}}body{{padding-block:12px}}</style>"
    return page.replace("</head>", style + "</head>", 1)


def only_chart(page: str, chart_id: str) -> str:
    """Show just one chart's figure card (its eyebrow, svg and caption)."""
    style = (
        "<style>main>*{display:none!important}"
        f"main>section:has(#{chart_id}){{display:grid!important}}"
        f"main>section:has(#{chart_id})>*:not(.pair:has(#{chart_id})):not(h2):not(div:has(>h2)){{display:none!important}}"
        f".pair:has(#{chart_id})>.fig:not(:has(#{chart_id})){{display:none!important}}"
        ".pair{grid-template-columns:1fr!important}body{padding-block:12px}</style>"
    )
    return page.replace("</head>", style + "</head>", 1)


DATA_DUMP = (
    "const __dump={};for(const b of [4,64]){bf=b;__dump[b]=strategies();}bf=4;renderAll();"
    "const __pre=document.createElement('pre');__pre.id='chartdata';__pre.textContent=JSON.stringify(__dump);"
    "document.body.appendChild(__pre);})();"
)


def shoot(page: str, name: str, width: int = 1200, height: int = 2400) -> Path:
    f = WORK / f"{name}.html"
    f.write_text(page, encoding="utf-8")
    png = WORK / f"{name}.png"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            "--virtual-time-budget=8000",
            f"--window-size={width},{height}",
            f"--screenshot={png}",
            f.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return png


def trim(png: Path, dest: Path, pad: int = 24) -> None:
    im = Image.open(png).convert("RGB")
    bg = Image.new("RGB", im.size, im.getpixel((2, 2)))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        im = im.crop(
            (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
        )
    im.save(dest, optimize=True)


made: list[str] = []
# Full page, both branching factors (tall window so nothing is cut; trimmed after).
for bf in (4, 64):
    png = shoot(with_bf(html, bf), f"full-bf{bf}", 1200, 7000)
    trim(png, OUT / f"full-page-bf{bf}.png")
    made.append(f"full-page-bf{bf}.png")
# Sections at bf=4; the five-metrics section also at bf=64.
for name, css in SECTIONS:
    trim(shoot(only(html, css), name), OUT / f"{name}.png")
    made.append(f"{name}.png")
trim(
    shoot(only(with_bf(html, 64), "section:nth-of-type(6)"), "06-five-metrics-bf64"),
    OUT / "06-five-metrics-bf64.png",
)
made.append("06-five-metrics-bf64.png")
# Single charts; the five metric charts at both branching factors.
for cid in CHARTS:
    trim(
        shoot(only_chart(html, cid), f"chart-{cid}-bf4", 760, 1400),
        OUT / f"chart-{CHART_NAMES[cid]}-bf4.png",
    )
    made.append(f"chart-{CHART_NAMES[cid]}-bf4.png")
    if cid in ("chartMult", "chartLat", "chartPly", "chartNeed", "chartActs"):
        trim(
            shoot(only_chart(with_bf(html, 64), cid), f"chart-{cid}-bf64", 760, 1400),
            OUT / f"chart-{CHART_NAMES[cid]}-bf64.png",
        )
        made.append(f"chart-{CHART_NAMES[cid]}-bf64.png")

# Data: patch a JSON dump of strategies() for both branching factors into the page, dump the DOM.
patched = html.replace("renderAll();\n})();", DATA_DUMP, 1)
assert patched != html, "data-dump anchor not found"
f = WORK / "data.html"
f.write_text(patched, encoding="utf-8")
dom = subprocess.run(
    [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--virtual-time-budget=8000",
        "--dump-dom",
        f.resolve().as_uri(),
    ],
    check=True,
    capture_output=True,
    timeout=120,
).stdout.decode("utf-8", "replace")
m = re.search(r'<pre id="chartdata">(.*?)</pre>', dom, re.S)
assert m, "chart data not found in DOM"
data = json.loads(
    m.group(1)
    .replace("&quot;", '"')
    .replace("&amp;", "&")
    .replace("&lt;", "<")
    .replace("&gt;", ">")
)
(OUT / "chart-data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
GAME_ACTIONS, GPU_MIN, CPU_MIN = 800, 532 * 10 / 110, 540 * 48 / 110
for bf, rows in data.items():
    with (OUT / f"chart-data-bf{bf}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "strategy",
                "family",
                "hardware",
                "seconds_per_action",
                "simulated_plies",
                "action_multiple_low",
                "action_multiple_high",
                "compute_minutes_per_game",
                "actions_affordable_per_game",
                "seconds_note",
                "plies_note",
                "multiple_note",
            ]
        )
        for s in rows:
            mult = s.get("mult")
            mid = (mult[0] + mult[1]) / 2 if mult else 1.0
            need = s["sec"] * GAME_ACTIONS * mid / 60 if s["hw"] != "brain" else ""
            avail = GPU_MIN if str(s["hw"]).startswith("GPU") else CPU_MIN
            acts = avail * 60 / s["sec"] if s["hw"] != "brain" else ""
            w.writerow(
                [
                    s["label"],
                    s["fam"],
                    s["hw"],
                    s["sec"],
                    s["plies"],
                    mult[0] if mult else "",
                    mult[1] if mult else "",
                    need,
                    acts,
                    s["secNote"],
                    s["pliesNote"],
                    s["multNote"],
                ]
            )
made += ["chart-data.json", "chart-data-bf4.csv", "chart-data-bf64.csv"]
print("\n".join(made))
