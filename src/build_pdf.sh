#!/usr/bin/env bash
# Compile a submission markdown file to PDF.
#
#   ./src/build_pdf.sh submissionv4.md submissionv4.pdf
#
# Requires pandoc + xelatex. Two markdown-to-LaTeX fixups happen here rather than in the
# source document, so the .md stays readable on its own:
#   1. the manual title block is replaced by a YAML metadata block;
#   2. the italic "*Figure N. ...*" paragraph after each image is folded into the image's alt
#      text, so pandoc emits it as a real figure caption instead of a floating paragraph.
set -euo pipefail

SRC="${1:-submissionv4.md}"
OUT="${2:-${SRC%.md}.pdf}"
# Page-fit knobs. Defaults reproduce every PDF built before these were added; override them when a
# submission has a hard page limit, e.g. MARGIN=0.85in PARSKIP=0.35em ./src/build_pdf.sh ...
MARGIN="${MARGIN:-0.95in}"
FONTSIZE="${FONTSIZE:-10pt}"
PARSKIP="${PARSKIP:-0.5em}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

python3 - "$REPO/$SRC" "$BUILD/paper.md" <<'PY'
import re, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_text()
# Take the title from the document's own H1 rather than hardcoding it here -- the title changed
# once already when the thesis moved, and a stale title on the PDF is invisible in the .md.
title = next(l[2:].strip() for l in raw.splitlines() if l.startswith('# '))
body = raw[raw.index('## Abstract'):]

paras, out, i = body.split('\n\n'), [], 0
while i < len(paras):
    # The trailing attribute block is optional: "![Fig 1](x.png)" and "![Fig 1](x.png){width=70%}"
    # both match. Without that alternative the second form silently fell through as raw markdown,
    # so pandoc emitted the alt text as the caption and left the real caption as a loose paragraph.
    img = re.match(r'^!\[(.*?)\]\(([^)]*)\)\s*(\{[^}]*\})?\s*$', ' '.join(paras[i].split()))
    if img and i + 1 < len(paras):
        cap = re.match(r'^\*(Figure [A-Z]?\d+\..*)\*$', ' '.join(paras[i + 1].split()), re.S)
        if cap:
            out.append(f'![{cap.group(1)}]({img.group(2)}){img.group(3) or "{width=100%}"}')
            i += 2
            continue
    out.append(paras[i])
    i += 1
body = '\n\n'.join(out)

# Keep the last appendix code block off its own orphan page.
for anchor in ('### A.6 Reproducing everything', '### A.4 Reproducing the figures'):
    if anchor in body:
        body = body.replace(anchor, '\\needspace{13\\baselineskip}\n\n' + anchor, 1)
        break

dst.write_text(f'''---
title: "{title}"
author:
  - Georg
  - Jonas
date: "Apart Research --- Secret Loyalties Hackathon, 2026"
---

''' + body)
PY

cat > "$BUILD/head.tex" <<EOF
\usepackage{geometry}
\geometry{a4paper,margin=$MARGIN}
\usepackage{caption}
\captionsetup{font=small,labelformat=empty,justification=raggedright,singlelinecheck=false,skip=6pt}
\usepackage{microtype}
\usepackage{needspace}
\setlength{\parskip}{$PARSKIP}
\setlength{\parindent}{0pt}
\usepackage{placeins}
% Figures are captioned inline in the source and read as part of the argument, so pin them where
% they are written. Letting them float leaves 20-line gaps on the pages they were pulled off.
\usepackage{float}
\floatplacement{figure}{H}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.72}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.7}
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1.0em}{0.35em}
\titlespacing*{\subsection}{0pt}{0.85em}{0.28em}
\raggedbottom
EOF

pandoc "$BUILD/paper.md" \
    --pdf-engine=xelatex \
    --resource-path="$REPO" \
    -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
    -V fontsize=$FONTSIZE -V colorlinks=true -V linkcolor=black -V urlcolor=blue \
    -H "$BUILD/head.tex" \
    -o "$REPO/$OUT"

echo "wrote $OUT ($(pdfinfo "$REPO/$OUT" | awk '/^Pages/{print $2}') pages)"
