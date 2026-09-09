# Frontier Marketing Measurement Study

A comprehensive survey of the current state and open frontiers of marketing measurement science, compiled August 2026 from recent papers, white papers, books, arXiv preprints, and big-tech publications (2020–2026 emphasis, foundational work included).

## Contents

| File | What it is |
|---|---|
| `01_state_of_the_field.md` | The study: current capabilities across (I) engineering for scale, (II) human understanding & marketing wisdom, (III) deep mathematics & statistics — with an executive synthesis of how the field converged on experiment-anchored triangulation. |
| `02_open_questions.md` | The frontier: 10 engineering, 10 human-side, and 11 mathematical open problems, each with why it's open, what a solution looks like, and who's closest — plus three cross-cutting grand challenges. |
| `03_mmm_adoption_barriers.md` | Why the majority of companies haven't adopted sophisticated MMM: 8 business-logic barriers (unit economics, incentives, precision theater), 10 mathematical barriers (weak identification, validation impossibility, prior elicitation), 7 operational barriers (data archaeology, geo-data, talent, experiment ops) — with adoption statistics and a tiered synthesis. |
| `resources/engineering_scale.md` | ~65 annotated sources: experimentation platforms, lift/ghost-ads/budget-split designs, geo experiments, open-source MMM, privacy-preserving measurement, marketplace designs, simulation. |
| `resources/human_wisdom.md` | ~60 annotated sources: Ehrenberg-Bass laws & critiques, long/short-term effects, elasticity meta-analyses, attention & creative measurement, synthetic consumers (LLMs), measurement politics, B2B. |
| `resources/math_stats.md` | ~66 annotated sources: causal inference core, MMM identification & calibration, experiment-vs-observational results, interference & auctions, attribution theory, adaptive experimentation, advanced identification. |
| `resources/mmm_adoption_barriers.md` | ~60 annotated sources behind the adoption-barriers study: industry surveys (Kantar/Meta, HBR/Google, IAB, EMARKETER), vendor field reports, practitioner critiques, cost data. |
| `resources/repos_and_tools.md` | The working codebases: Meridian, Robyn, PyMC-Marketing, GeoLift, Trimmed Match, EconML/grf/CausalML, AuctionGym, genagents, plus datasets and standards. |
| `resources/books_and_institutes.md` | Bookshelf, institutes, journals, arXiv feeds, and people whose output tracks the frontier. |

## How to read it

Start with the executive synthesis in `01_state_of_the_field.md` (one page), then go straight to `02_open_questions.md` — the three cross-cutting grand challenges at the end are the highest-leverage summary of where the science has not yet been developed. The resource files are designed to be grep-able: every entry has title, authors, year, one-line significance, and a direct link (PDF where freely available).

## Provenance note

Compiled via parallel deep literature surveys (web search + source fetching) across the three focus areas. PDF links were taken from search results and fetched pages rather than fabricated; a handful of very recent arXiv IDs (2026) are worth re-verifying as those preprints evolve. PDFs are linked rather than vendored into the repo.
