#!/usr/bin/env bash
# Full build of the compendium. Usage: ./build.sh
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p build/full
# non-ASCII check across all inputs
if LC_ALL=C grep -nP '[^\x00-\x7F]' compendium.tex preamble.tex frontmatter.tex intro.tex appendices.tex ledger_table.tex chapters/ch*.tex; then
  echo "ERROR: non-ASCII characters found (see above)"; exit 2
fi
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -file-line-error -output-directory=build/full compendium.tex > build/full/pass$pass.out 2>&1
  if grep -qE '^! ' build/full/pass$pass.out; then
    echo "LaTeX ERROR (pass $pass):"; grep -nE '^.*:[0-9]+: |^! ' build/full/pass$pass.out | head -20; exit 1
  fi
done
echo "Pages: $(grep -oE 'Output written on .*\(([0-9]+) pages' build/full/pass3.out | grep -oE '[0-9]+ pages')"
echo "Overfull hboxes (>30pt): $(grep -cE 'Overfull \\hbox \(([3-9][0-9]|[0-9]{3,})' build/full/compendium.log)"
grep -E 'Overfull \\hbox \(([3-9][0-9]|[0-9]{3,})' build/full/compendium.log | head -12
echo "Undefined references: $(grep -c 'Reference .* undefined' build/full/compendium.log)"
grep 'Reference .* undefined' build/full/compendium.log | head -10
echo "Multiply-defined labels: $(grep -c 'multiply defined' build/full/compendium.log)"
grep 'multiply defined' build/full/compendium.log | head
echo "Citation/label warnings: $(grep -c 'LaTeX Warning: Label' build/full/compendium.log)"
