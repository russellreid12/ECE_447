# Train ISTA

## 1) Install dependencies

From the project root:

```bash
python3 -m pip install -r requirements.txt
```

## 2) Generate MNIST tensors (first run only)

This creates preprocessed tensors in the default `mnist/` directory.

```bash
python3 main.py --task generate
```

## 3) Train ISTA

Run ISTA training and evaluation on MNIST:

```bash
python3 main.py --task train
```

## 4) Common ISTA options

```bash
python3 main.py --task train \
  --output-dir mnist \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --lambda-reg 1e-3 \
  --max-iter 300 \
  --tol 1e-5
```

## 5) Useful variants

Use all available training and test samples:

```bash
python3 main.py --task train --max-train-samples -1 --max-test-samples -1
```

Run quietly (less console output):

```bash
python3 main.py --task train --quiet
```

Generate data and then train in one command:

```bash
python3 main.py --task all
```

## 6) Parameter quick reference

- `--lambda-reg`: L1 regularization strength for sparsity.
- `--max-iter`: Maximum ISTA iterations.
- `--tol`: Relative change tolerance for convergence.
- `--max-train-samples`: Number of train samples (`<= 0` means all).
- `--max-test-samples`: Number of test samples (`<= 0` means all).
- `--output-dir`: Directory containing MNIST tensor files (`x_train.pt`, `y_train.pt`, `x_test.pt`, `y_test.pt`).
