# Train LISTA

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

## 3) Train LISTA

Run LISTA training and evaluation on MNIST:

```bash
python3 main.py --task train-lista
```

## 4) Common LISTA options

```bash
python3 main.py --task train-lista \
  --output-dir mnist \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --lambda-reg 1e-3 \
  --lista-layers 5 \
  --lista-code-dim 256 \
  --lista-epochs 10 \
  --lista-batch-size 128 \
  --lista-lr 1e-3
```

## 5) Checkpoint workflow

Train and save a checkpoint:

```bash
python3 main.py --task train-lista --save-lista-checkpoint
```

Train while loading an existing checkpoint:

```bash
python3 main.py --task train-lista --load-lista-checkpoint
```

Use a custom checkpoint path:

```bash
python3 main.py --task train-lista \
  --lista-checkpoint-path checkpoints/my_lista.pt \
  --save-lista-checkpoint
```

Evaluate a saved LISTA checkpoint without further training:

```bash
python3 main.py --task train-lista --load-lista-checkpoint --lista-epochs 0
```

## 6) Useful variants

Use all available training and test samples:

```bash
python3 main.py --task train-lista --max-train-samples -1 --max-test-samples -1
```

Run quietly (less console output):

```bash
python3 main.py --task train-lista --quiet
```

Generate data and then run LISTA in one command:

```bash
python3 main.py --task all-lista
```

## 7) Parameter quick reference

- `--lambda-reg`: L1 sparsity penalty weight on the LISTA code.
- `--lista-layers`: Number of unfolded LISTA layers.
- `--lista-code-dim`: Sparse code dimension.
- `--lista-epochs`: Number of training epochs (`<= 0` skips training loop).
- `--lista-batch-size`: Batch size for LISTA training.
- `--lista-lr`: Adam learning rate.
- `--load-lista-checkpoint`: Load model + scaler from checkpoint before run.
- `--save-lista-checkpoint`: Save model + scaler after run.
- `--lista-checkpoint-path`: Checkpoint file location.
- `--output-dir`: Directory containing MNIST tensor files (`x_train.pt`, `y_train.pt`, `x_test.pt`, `y_test.pt`).
