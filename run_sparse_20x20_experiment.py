from __future__ import annotations

from pathlib import Path

from experiment.benchmark import benchmark_sparse_20x20


def main() -> None:
    benchmark_sparse_20x20(
        plots_dir=Path("plots"),
        lambda_reg=1e-3,
        max_iter=7,
        lista_layers=7,
        lista_code_dim=100,
        lista_epochs=10,
        lista_batch_size=128,
        lista_lr=1e-3,
        max_train_samples=50000,
        max_test_samples=5000,
    )


if __name__ == "__main__":
    main()
