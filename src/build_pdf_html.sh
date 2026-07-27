#!/usr/bin/env bash
# Compile a submission markdown file to PDF without LaTeX.
#
#   ./src/build_pdf_html.sh submissionv6.md submissionv6.pdf
#
# build_pdf.sh is the preferred path and produces better typography, but it needs xelatex, which is
# not installed on every machine this repo gets checked out on. This route is pandoc -> standalone
# HTML with the figures embedded as data URIs -> headless Chrome print-to-PDF, which needs only
# pandoc and a Chrome that is already there.
#
# Same two markdown fixups as build_pdf.sh: the manual title block becomes real metadata, and the
# italic "*Figure N. ...*" paragraph after each image becomes that figure's caption.
set -euo pipefail

SRC="${1:-submissionv6.md}"
OUT="${2:-${SRC%.md}.pdf}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

python3 - "$REPO/$SRC" "$BUILD/paper.md" <<'PY'
import re, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_text()
title = next(l[2:].strip() for l in raw.splitlines() if l.startswith('# '))
body = raw[raw.index('## Abstract'):]

paras, out, i = body.split('\n\n'), [], 0
while i < len(paras):
    img = re.match(r'^!\[(.*?)\]\((.*?)\)\s*$', ' '.join(paras[i].split()))
    if img and i + 1 < len(paras):
        cap = re.match(r'^\*(Figure [A-Z]?\d+\..*)\*$', ' '.join(paras[i + 1].split()), re.S)
        if cap:
            out.append(f'<figure><img src="{img.group(2)}">'
                       f'<figcaption>{cap.group(1)}</figcaption></figure>')
            i += 2
            continue
    out.append(paras[i])
    i += 1
body = '\n\n'.join(out)

dst.write_text(f'''---
title: "{title}"
author:
  - Georg
  - Jonas
date: "Apart Research --- Secret Loyalties Hackathon, 2026"
---

''' + body)
PY

cat > "$BUILD/style.css" <<'EOF'
@page { size: A4; margin: 20mm 18mm; }
html { font-size: 10.5pt; }
body { font-family: "DejaVu Serif", Georgia, serif; line-height: 1.45; max-width: none;
       color: #111; margin: 0; }
h1.title { font-size: 1.55rem; line-height: 1.25; margin-bottom: 0.2em; }
p.author, p.date { margin: 0.15em 0; color: #444; font-size: 0.95rem; }
h2 { font-size: 1.18rem; margin: 1.3em 0 0.35em; border-bottom: 1px solid #ddd;
     padding-bottom: 0.15em; }
h3 { font-size: 1.02rem; margin: 1.1em 0 0.3em; }
p { margin: 0 0 0.6em; text-align: justify; hyphens: auto; }
table { border-collapse: collapse; font-size: 0.88rem; margin: 0.7em 0; width: 100%;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
th { background: #f4f4f4; }
figure { margin: 0.9em 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figcaption { font-size: 0.85rem; color: #333; text-align: left; margin-top: 0.35em; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 0.85em;
       background: #f4f4f4; padding: 0 2px; }
pre { background: #f6f6f6; padding: 8px 10px; font-size: 0.8rem; overflow-x: auto;
      page-break-inside: avoid; }
blockquote { margin: 0.7em 0; padding: 0.35em 0 0.35em 0.9em; border-left: 3px solid #999;
             color: #222; }
h2, h3 { page-break-after: avoid; }
EOF

pandoc "$BUILD/paper.md" \
    --standalone --embed-resources --resource-path="$REPO" \
    --css="$BUILD/style.css" --metadata=lang:en \
    -f markdown+raw_html -t html5 \
    -o "$BUILD/paper.html"

"$CHROME" --headless --disable-gpu --no-pdf-header-footer --no-sandbox \
    --print-to-pdf="$REPO/$OUT" "file://$BUILD/paper.html" >/dev/null 2>&1

python3 - "$REPO/$OUT" <<'PY'
import re, sys
from pathlib import Path
data = Path(sys.argv[1]).read_bytes()
pages = len(re.findall(rb'/Type\s*/Page[^s]', data))
print(f"wrote {sys.argv[1]} ({len(data)//1024} KB, {pages} pages)")
PY
