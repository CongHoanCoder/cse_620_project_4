# CSE 620 Mini-Project — Niching Genetic Algorithms

Compares three genetic algorithm variants on the multimodal benchmark functions
**M1** and **M4** (Mahfoud, ICGA 1995): a standard GA (no niching),
**deterministic crowding**, and **fitness sharing**.

## Layout

| Path | Description |
|------|-------------|
| `niching_ga_experiments.py` | Main experiment runner (bug-fixed/extended version of the course notebook) |
| `niching_ga_examplever.py` | Original course-provided notebook, kept for reference |
| `make_report_plots.py` | Builds every report figure from `results/experiments.json` |
| `results/experiments.json` | All 240 runs (2 functions × 3 methods × 2 Pc × 2 Pm × 10 seeds) |
| `report/` | `report.tex`, `IEEEtran.cls`, `figures/`, and the built `report.pdf` |
| `results_run.log` | Console output of the full protocol run |

## Requirements

- Python 3 with `numpy` and `matplotlib`
- A LaTeX distribution with `latexmk`/`pdflatex` (for the report)

## Usage

```bash
# Full protocol: 240 runs, 300 generations each -> results/experiments.json
python3 niching_ga_experiments.py

# Quick smoke test
python3 niching_ga_experiments.py --quick

# Custom run
python3 niching_ga_experiments.py --num-runs 10 --num-gen 300 --pop-size 50

# Regenerate all figures
python3 make_report_plots.py

# Build the report
cd report && latexmk -pdf report.tex
```

## Key results (average of 10 runs, peaks found out of 5)

| Function | Crowding | Sharing | Standard GA |
|----------|----------|---------|-------------|
| M1 (equal-height peaks) | **5.0 ± 0.0** | 3.8–4.2 | 1.1–3.2 |
| M4 (unequal-height peaks) | 1.9–2.3 | **3.5–3.9** | 1.1–2.5 |

Deterministic crowding dominates on M1 but drops the shorter peaks of M4;
fitness sharing is the most robust overall. Best fitness alone hides diversity
collapse — at low mutation rates the standard GA converges to a single peak.

## Protocol

10-bit binary encoding, two-point crossover, bit-flip mutation; population 50,
300 generations, `Pc ∈ {0.60, 0.95}`, `Pm ∈ {0.01, 0.10}`, 10 seeds per
configuration. Differences from the original notebook (both children inserted,
fixed fitness computation, non-mutating peak counting, 5% uniform selection
mix, generational replacement with elitism for the fitness-based methods) are
documented in the docstring of `niching_ga_experiments.py`.
