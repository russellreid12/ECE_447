# ECE 447 LISTA Project

Simple quick-start guide for running MNIST experiments.

## 1) Install dependencies

From the project root:

```bash
python3 -m pip install -r requirements.txt
```

## 2) Generate MNIST tensors

```bash
python3 main.py --task generate
```

This creates tensor files under `mnist/`.

## 3) Train ISTA

```bash
python3 main.py --task train
```

## 4) Train LISTA

```bash
python3 main.py --task train-lista
```

Example with custom settings:

```bash
python3 main.py --task train-lista \
  --lista-layers 3 \
  --lista-code-dim 256 \
  --lista-epochs 1000
```

## 5) Run benchmark plots

```bash
python3 main.py --task benchmark
```

Plots are saved to `plots/`.

## 6) Helpful docs

- ISTA guide: `docs/train_ista.md`
- LISTA guide: `docs/train_lista.md`
