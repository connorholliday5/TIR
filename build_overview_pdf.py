#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Renders OVERVIEW.md as a print-ready PDF handout.
#
#     python build_overview_pdf.py
#
# Chromium does the typesetting, so the result matches what a browser would
# print and needs no LaTeX toolchain.  Kept in the repository because the
# handout has to be regenerated whenever the overview changes, and a PDF
# rebuilt by hand drifts from the document it is supposed to mirror.

import glob
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "OVERVIEW.md"
OUTPUT = ROOT / "OVERVIEW.pdf"

# Links to sibling documents are useful on GitHub and meaningless on paper.
DROP = (
    "Companion documents: **[README.md](README.md)** to run it,\n"
    "**[ARCHITECTURE.md](ARCHITECTURE.md)** for how the code fits together,\n"
    "**[PEER_REVIEW.md](PEER_REVIEW.md)** to review it."
)

STYLE = """
  @page { size: letter; margin: 0.85in 0.8in 0.9in; }
  html { font-size: 10.5pt; }
  body { font-family: Georgia, "Times New Roman", serif; line-height: 1.5;
         color: #111; margin: 0;
         -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  h1 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 21pt;
       line-height: 1.15; margin: 0 0 0.15in; padding-bottom: 0.08in;
       border-bottom: 2.5pt solid #1F4E79; color: #12161C; }
  h2 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 13pt;
       margin: 0.32in 0 0.1in; color: #1F4E79; padding-bottom: 0.04in;
       border-bottom: 0.75pt solid #C8D2DE;
       break-after: avoid; page-break-after: avoid; }
  h3 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11pt;
       margin: 0.2in 0 0.06in; color: #12161C;
       break-after: avoid; page-break-after: avoid; }
  p { margin: 0 0 0.11in; orphans: 3; widows: 3; }
  strong { color: #000; }
  /* The tables carry most of the evidence; splitting one across a page break
     separates a figure from the field it belongs to. */
  table { border-collapse: collapse; width: 100%; margin: 0.1in 0 0.16in;
          font-size: 9.5pt; break-inside: avoid; page-break-inside: avoid; }
  th, td { text-align: left; padding: 4pt 8pt 4pt 0;
           border-bottom: 0.5pt solid #D8DEE6; vertical-align: top; }
  th { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8pt;
       text-transform: uppercase; letter-spacing: 0.04em; color: #4A5563;
       border-bottom: 1pt solid #9DAAB8; }
  td[align="right"], th[align="right"] { text-align: right; padding-right: 0;
       font-variant-numeric: tabular-nums; }
  ul, ol { margin: 0 0 0.13in; padding-left: 0.22in; }
  li { margin-bottom: 0.05in; }
  hr { border: 0; border-top: 0.5pt solid #D8DEE6; margin: 0.26in 0; }
  blockquote { margin: 0.12in 0; padding: 0.08in 0 0.08in 0.14in;
               border-left: 2.5pt solid #1F4E79; background: #F4F7FA;
               font-size: 9.5pt; }
  blockquote p:last-child { margin-bottom: 0; }
  code { font-family: "Consolas", "Courier New", monospace; font-size: 9pt;
         background: #F0F3F7; padding: 0 2pt; }
  em { color: #38424E; }
"""


# Finds the browser that will do the typesetting.
def find_chromium() -> str:
    """Return a Chromium executable, or exit saying none was found."""
    candidates = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")
    for name in ("chromium", "chromium-browser", "google-chrome"):
        found = subprocess.run(["which", name], capture_output=True, text=True)
        if found.returncode == 0:
            candidates.append(found.stdout.strip())
    if not candidates:
        sys.exit("No Chromium found. Install chromium, or print OVERVIEW.md from any browser.")
    return candidates[0]


def main() -> None:
    md = SOURCE.read_text().replace(DROP, "")
    md = re.sub(r"^\n# ", "# ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)

    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>TIR Liability Coding — Overview</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>"
    )

    page = ROOT / ".overview.print.html"
    page.write_text(html)
    try:
        subprocess.run([
            find_chromium(), "--headless", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer", f"--print-to-pdf={OUTPUT}", str(page),
        ], check=True, capture_output=True)
    finally:
        page.unlink(missing_ok=True)

    print(f"✔ Wrote {OUTPUT.name} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
