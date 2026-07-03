# Emergent Inequality from Equal Starts

Agent-based computational economics: LaTeX paper, Python simulations, and figures.

**Core claim:** inequality emerges from **unbiased** random wealth exchange when all agents start equal. The Matthew effect (bias toward the richer agent) is an **extension** that accelerates concentration—it is not required for inequality to appear.

## Quick start

```powershell
pip install -r requirements.txt
python simulation/generate_figures.py
python simulation/run_simulation.py --model dy
python simulation/run_simulation.py --model dy --ensemble 1000
```

## Models

| Act | Model | Bias toward rich? | Role |
|-----|-------|-------------------|------|
| I | Random exchange (DY) | No | **Core** — minimal proof |
| I | Saving propensity | No | Stronger tails, still unbiased |
| I | Yard-sale | No | Min-stake structure, still unbiased |
| II | Matthew effect | Yes ($p > 1/2$) | **Extension** — amplification |

## Key figure

`fig_ensemble_unbiased` — 1,000 independent unbiased runs, all starting at Gini = 0, all ending with Gini > 0.

## Compile paper

```powershell
cd paper
pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
```

Output: `paper/main.pdf`

## Repository

[Agent-Based-Wealth-Exchange-and-the-Matthew-Effect-Computational-Economics-](https://github.com/sbf-developer/Agent-Based-Wealth-Exchange-and-the-Matthew-Effect-Computational-Economics-)
