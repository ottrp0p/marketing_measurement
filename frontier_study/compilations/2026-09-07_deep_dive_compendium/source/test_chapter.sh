#!/usr/bin/env bash
# Standalone compile test for one chapter file.
# Usage: ./test_chapter.sh chapters/ch03_voi.tex
# Exit status 0 = compiled without LaTeX errors; prints warnings summary.
set -u
CH="$1"
ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build/test_$(basename "$CH" .tex)"
mkdir -p "$BUILD"

# 1) Non-ASCII check (pdflatex chokes on unicode math symbols)
if LC_ALL=C grep -nP '[^\x00-\x7F]' "$CH" > "$BUILD/nonascii.txt"; then
  echo "ERROR: non-ASCII characters found in $CH (replace with LaTeX commands):"
  head -20 "$BUILD/nonascii.txt"
  exit 2
fi

# 2) Forbidden constructs
if grep -vE '^\s*%' "$CH" | grep -nE '\\usepackage|\\begin\{document\}|\\documentclass'; then
  echo "ERROR: chapter files must not load packages or start a document."
  exit 2
fi

# 3) Wrapper
cat > "$BUILD/wrap.tex" <<EOF
\documentclass[10pt,letterpaper,oneside]{report}
\input{$ROOT/preamble.tex}
\begin{document}
\setcounter{chapter}{0}
\input{$ROOT/$CH}
\end{document}
EOF

cd "$BUILD"
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error wrap.tex > pdflatex.out 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "LaTeX ERROR in $CH (pass $pass):"
    grep -nE '^.*:[0-9]+: |^! ' pdflatex.out | head -20
    grep -A3 -E '^! ' pdflatex.out | head -40
    exit 1
  fi
done

PAGES=$(grep -oE 'Output written on wrap.pdf \([0-9]+ pages?' pdflatex.out | grep -oE '[0-9]+ pages?')
echo "OK: $CH compiled ($PAGES)."
echo "Overfull hboxes: $(grep -c 'Overfull \\hbox' wrap.log)"
grep -E 'Overfull \\hbox \((1[0-9][0-9]|[2-9][0-9][0-9])' wrap.log | head -10
echo "Undefined references: $(grep -c 'Reference .* undefined' wrap.log)"
grep 'Reference .* undefined' wrap.log | head -10
exit 0
