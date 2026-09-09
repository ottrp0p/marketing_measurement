# Deep-Dive Compendium — compiled 2026-09-07

**`compendium_2026-09-07.pdf`** — *New Mathematics for Marketing Measurement: A Compendium of the Deep-Dive Program*, 186 pages, letter, print-ready.

This is a compilation document: it stitches the deep-dive runs 01–22 (31 Aug – 7 Sep 2026) in `../../deep_dives/` into fifteen self-contained chapters, one per dive (extension pairs 03+04, 08+09, 13+14, 16+19 merged; the identification thread 01+20+21+22 merged into Chapter 1). Nothing from run 23 onward is included.

## What is in the document

- Front matter: what was compiled, how chapters are organised, a legend for the epistemic tags, and a **provenance index** (Table 1) listing every source dive with its run date from the program log, the report file's last-written timestamp on disk at compilation, word count and chapter.
- Introduction: the frontier study's framing, a chapter map with dependencies, and the ideas that recur across dives.
- Chapters 1–15, each with: provenance, key ideas, problem and motivation, background (the referenced papers' key ideas reproduced in summary), setup and assumptions, main results (with the dive's own labels: established / transported / numerical / conjecture / refuted), numerical evidence tables, limitations and devil's advocate, discussion of impacts, further exploration, compiler's reviewer notes, and sources.
- Appendix A: the open-problem ledger (all 88 BL items, who opened them, status at compilation, chapter).
- Appendix B: the program's protocol. Appendix C: file inventory.

## Sources (as on disk 2026-09-07)

`frontier_study/deep_dives/00_INDEX.md`, `01_*.md` … `22_*.md`, `deep_dives/code/*`, the framing documents `01_state_of_the_field.md`, `02_open_questions.md`, `03_mmm_adoption_barriers.md`, and `resources/*.md`. Timestamps are in Table 1 of the PDF.

## How it was built

`source/` holds the LaTeX: `compendium.tex` (master), `preamble.tex`, `frontmatter.tex`, `intro.tex`, `appendices.tex`, `ledger_table.tex` (generated from the INDEX backlog by `make_ledger.py`), and `chapters/ch01…ch15.tex`. Rebuild with `./build.sh` (pdflatex, TeX Live 2023; Palatino via `mathpazo`). `test_chapter.sh` compiles a single chapter standalone.

Each chapter was drafted from its dive(s) by an author agent under a fixed brief, then independently verified against the dive markdown and code by a second agent (headline-result coverage, 12+ number spot-checks, label/tag hygiene). The orange "Reviewer's note" boxes are the only content that does not come from the dives. No simulation was re-run.
