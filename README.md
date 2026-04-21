# ECE 447 LISTA Reproduction

This project reproduces the LISTA experiments from Gregor and LeCun (2010) and compares the learned sparse-coding model against FISTA on the MNIST-based setup used in the paper.

## How It Works

- Loads the MNIST data and prepares the sparse-coding experiments used in the paper.
- Trains a LISTA model to compare against FISTA on the same problem setup.
- Saves the resulting figures, tables, and cached experiment outputs in the `results/` folder.

See the paper [here](./gregor-2010-lista.pdf) for more details.

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

### Run Experiment From Cache (Pre-trained Model)

Run from the project root:

```bash
python lista.py
```

- This uses the existing model that has been trained in `results/cache/experiment_results.pkl`
- This is done because training the model takes a long time (up to 1 hour depending on computer speed)
- To run from scratch, see the next section

### Run Experiment From Scratch

Run from the project root:
```bash
python lista.py --clean
```

The script writes outputs to:
- `results/cache/experiment_results.pkl`
- `results/figures/*.png`
- `results/tables/table1_results.txt`
- `results/tables/table1_results.csv`

- NOTE: This takes a long time to execute!

## Project Layout

```
project/
├── lista.py                 # Main experiment/training script
├── PROJECT_REPORT.pdf       # Written report for the project
├── data/                    # MNIST dataset storage
└── results/
	├── cache/               # Serialized experiment artifacts (.pkl)
	├── figures/             # Generated PNG plots
	└── tables/              # Generated table outputs (.txt/.csv)
```

## References

- Python implementation of the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA): https://github.com/jeankossaifi/fista



