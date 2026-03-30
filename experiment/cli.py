from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import benchmark_performance
from .data import generate_mnist
from .training import train_ista, train_lista


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST generation and ISTA/LISTA experiments.")
    parser.add_argument(
        "--task",
        choices=["generate", "train", "all", "train-lista", "all-lista", "benchmark"],
        default="all",
        help="Task to run: generate MNIST tensors, train ISTA/LISTA model, or run benchmark graphs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mnist"),
        help="Directory to save processed tensors (.pt files).",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for downloaded torchvision MNIST raw files.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Number of train samples for training (<=0 means use all).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=2000,
        help="Number of test samples for evaluation (<=0 means use all).",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=1e-3,
        help="L1 regularization strength.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum ISTA iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Relative change tolerance for ISTA convergence.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable training logs.",
    )
    parser.add_argument(
        "--lista-layers",
        type=int,
        default=5,
        help="Number of unfolded LISTA layers.",
    )
    parser.add_argument(
        "--lista-code-dim",
        type=int,
        default=256,
        help="Latent sparse code dimension for LISTA classifier.",
    )
    parser.add_argument(
        "--lista-epochs",
        type=int,
        default=10,
        help="Training epochs for LISTA.",
    )
    parser.add_argument(
        "--lista-batch-size",
        type=int,
        default=128,
        help="Batch size for LISTA training.",
    )
    parser.add_argument(
        "--lista-lr",
        type=float,
        default=1e-3,
        help="Learning rate for LISTA training.",
    )
    parser.add_argument(
        "--lista-checkpoint-path",
        type=Path,
        default=Path("checkpoints/lista_checkpoint.pt"),
        help="Path to save/load LISTA checkpoint.",
    )
    parser.add_argument(
        "--load-lista-checkpoint",
        action="store_true",
        help="Load LISTA model + scaler from checkpoint before training/evaluation.",
    )
    parser.add_argument(
        "--save-lista-checkpoint",
        action="store_true",
        help="Save LISTA model + scaler after run.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save benchmark graph images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.task in {"generate", "all", "all-lista"}:
        generate_mnist(output_dir=args.output_dir, raw_data_dir=args.raw_data_dir)

    if args.task in {"train", "all"}:
        train_ista(
            mnist_dir=args.output_dir,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=not args.quiet,
        )

    if args.task in {"train-lista", "all-lista"}:
        train_lista(
            mnist_dir=args.output_dir,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            lambda_reg=args.lambda_reg,
            lista_layers=args.lista_layers,
            lista_code_dim=args.lista_code_dim,
            lista_epochs=args.lista_epochs,
            lista_batch_size=args.lista_batch_size,
            lista_lr=args.lista_lr,
            lista_checkpoint_path=args.lista_checkpoint_path,
            load_lista_checkpoint=args.load_lista_checkpoint,
            save_lista_checkpoint=args.save_lista_checkpoint,
            verbose=not args.quiet,
        )

    if args.task == "benchmark":
        benchmark_performance(
            mnist_dir=args.output_dir,
            raw_data_dir=args.raw_data_dir,
            plots_dir=args.plots_dir,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            tol=args.tol,
            lista_layers=args.lista_layers,
            lista_code_dim=args.lista_code_dim,
            lista_epochs=args.lista_epochs,
            lista_batch_size=args.lista_batch_size,
            lista_lr=args.lista_lr,
        )
