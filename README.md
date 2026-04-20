# ECE 447 LISTA Reproduction

Reproduction of Gregor and LeCun (2010), including:
- Figure 3 comparisons of FISTA vs LISTA
- Table 1 code prediction error for dictionary sizes m=100 and m=400

## Project Layout

```
project/
├── lista.py                 # Main experiment/training script
├── data/                    # MNIST dataset storage
└── results/
	├── cache/               # Serialized experiment artifacts (.pkl)
	├── figures/             # Generated PNG plots
	└── tables/              # Generated table outputs (.txt/.csv)
```

## Getting Started

### Conda Environment Setup

Create and activate a dedicated conda environment:

```bash
conda create -n ece447-lista python=3.11 -y
conda activate ece447-lista
```

Install project dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Running Experiments

Run from the project root:

```bash
python lista.py
```

The script writes outputs to:
- `results/cache/experiment_results.pkl`
- `results/figures/*.png`
- `results/tables/table1_results.txt`
- `results/tables/table1_results.csv`

## Notes

- If `results/cache/experiment_results.pkl` exists, the script reuses cached results.
- Remove that file if you want a full retrain and rerun.
- Training from scratch takes a long time (usually >1 hour)
