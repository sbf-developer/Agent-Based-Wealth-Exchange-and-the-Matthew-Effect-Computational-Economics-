# Emergent Inequality and the Matthew Effect

An agent-based computational economics project: PhD-level working paper (LaTeX), Python wealth-exchange simulations, and publication figures.

## What this is

When agents start with **identical wealth** and interact through random pairwise exchange, **inequality emerges endogenously**. This is sometimes called:

- **Agent-based wealth distribution model**
- **Emergent inequality** / **wealth condensation**
- The **Matthew effect** (cumulative advantage: "to those who have, more will be given")

## Project structure

```
Matthew Effect/
├── paper/
│   ├── main.tex           # LaTeX paper (opens with Matthew 25:29)
│   ├── references.bib     # Academic references
│   └── figures/           # Generated PDF/PNG figures
├── simulation/
│   ├── models.py          # Four exchange models + metrics
│   ├── generate_figures.py  # Build all paper figures
│   └── run_simulation.py  # CLI for single runs
└── requirements.txt
```

## Quick start

```powershell
pip install -r requirements.txt
python simulation/generate_figures.py
python simulation/run_simulation.py --model matthew
```

## Compile the paper

Requires a LaTeX distribution (TeX Live, MiKTeX):

```powershell
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `paper/main.pdf`

## Models implemented

| Model | Reference | Mechanism |
|-------|-----------|-----------|
| Random exchange (DY) | Dragulescu & Yakovenko (2000) | Fixed stake, fair coin flip |
| Saving propensity | Chakraborti & Chakrabarti (2000) | Agents save fraction λ before trading |
| Yard-sale | Ispolatov et al. (1998) | Loser pays fraction of own wealth |
| Matthew effect | Merton (1968) | Transfers biased toward richer agent |

All models conserve total wealth. Initial conditions: \(w_i(0) = 1\) for all agents.

## Figures

1. Gini evolution over time (all models)
2. Wealth distribution CCDFs (log-log)
3. Lorenz curves
4. Matthew-effect heatmap (Gini vs bias and stake)
5. Top 1% wealth share dynamics
6. Histogram evolution (equality → concentration)
7. Saving propensity comparative statics

## Citation

If you use this code, please cite the underlying literature (Merton 1968; Dragulescu & Yakovenko 2000; Ispolatov et al. 1998; Yakovenko & Rosser 2009).
