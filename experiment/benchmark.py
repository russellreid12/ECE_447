from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_mnist, get_preprocessed_mnist_arrays
from .models import ISTAClassifier, LISTASparseCoder
from .training import train_ista, train_lista


def benchmark_performance(
    mnist_dir: Path,
    raw_data_dir: Path,
    plots_dir: Path,
    max_train_samples: int,
    max_test_samples: int,
    lambda_reg: float,
    max_iter: int,
    tol: float,
    lista_layers: int,
    lista_code_dim: int,
    lista_epochs: int,
    lista_batch_size: int,
    lista_lr: float,
) -> None:
    required_files = ["x_train.pt", "y_train.pt", "x_test.pt", "y_test.pt"]
    if not all((mnist_dir / name).exists() for name in required_files):
        print("MNIST tensors not found; generating dataset first...")
        generate_mnist(output_dir=mnist_dir, raw_data_dir=raw_data_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)

    ista_results = train_ista(
        mnist_dir=mnist_dir,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        lambda_reg=lambda_reg,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    lista_results = train_lista(
        mnist_dir=mnist_dir,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        lambda_reg=lambda_reg,
        lista_layers=lista_layers,
        lista_code_dim=lista_code_dim,
        lista_epochs=lista_epochs,
        lista_batch_size=lista_batch_size,
        lista_lr=lista_lr,
        lista_checkpoint_path=Path("checkpoints/lista_checkpoint.pt"),
        load_lista_checkpoint=False,
        save_lista_checkpoint=False,
        verbose=False,
    )

    if ista_results is None or lista_results is None:
        raise RuntimeError("Benchmark failed to produce training results.")

    ista_history = cast(list[dict[str, float]], ista_results["history"])
    lista_history = cast(list[dict[str, float]], lista_results["history"])

    ista_iterations = np.array([item["iteration"] for item in ista_history], dtype=np.float32)
    ista_objective = np.array([item["objective"] for item in ista_history], dtype=np.float32)

    lista_epochs_axis = np.array([item["epoch"] for item in lista_history], dtype=np.float32)
    lista_train_loss = np.array([item["train_loss"] for item in lista_history], dtype=np.float32)
    lista_test_acc = np.array([item["test_accuracy"] for item in lista_history], dtype=np.float32)
    lista_elapsed = np.array([item["elapsed_sec"] for item in lista_history], dtype=np.float32)

    checkpoint_count = min(12, max_iter)
    ista_checkpoints = sorted({
        int(value)
        for value in np.linspace(1, max_iter, num=checkpoint_count, dtype=int).tolist()
        if value >= 1
    })

    x_train, y_train, x_test, y_test = get_preprocessed_mnist_arrays(
        mnist_dir=mnist_dir,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )

    ista_checkpoint_acc: list[float] = []
    ista_checkpoint_time: list[float] = []
    for checkpoint_iter in ista_checkpoints:
        checkpoint_model = ISTAClassifier(
            lambda_reg=lambda_reg,
            max_iter=checkpoint_iter,
            tol=tol,
            verbose=False,
        )
        checkpoint_model.fit(x_train, y_train, num_classes=10)
        checkpoint_pred = checkpoint_model.predict(x_test)
        checkpoint_acc = accuracy_score(y_test, checkpoint_pred)
        ista_checkpoint_acc.append(float(checkpoint_acc))
        ista_checkpoint_time.append(float(checkpoint_model.history[-1]["elapsed_sec"]))

    plt.figure(figsize=(7, 4.5))
    plt.plot(ista_iterations, ista_objective, label="ISTA objective", linewidth=2)
    plt.xlabel("ISTA iteration")
    plt.ylabel("Objective")
    plt.title("ISTA convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ista_objective_vs_iteration.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(lista_epochs_axis, lista_train_loss, label="LISTA train loss", linewidth=2)
    plt.xlabel("LISTA epoch")
    plt.ylabel("Loss")
    plt.title("LISTA training curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "lista_loss_vs_epoch.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(ista_checkpoints, ista_checkpoint_acc, marker="o", label="ISTA test accuracy")
    plt.plot(lista_epochs_axis, lista_test_acc, marker="s", label="LISTA test accuracy")
    plt.xlabel("Optimization step (iteration/epoch)")
    plt.ylabel("Test accuracy")
    plt.title("ISTA vs LISTA accuracy progression")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ista_vs_lista_accuracy_progression.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(ista_checkpoint_time, ista_checkpoint_acc, marker="o", label="ISTA")
    plt.plot(lista_elapsed, lista_test_acc, marker="s", label="LISTA")
    plt.xlabel("Elapsed time (seconds)")
    plt.ylabel("Test accuracy")
    plt.title("ISTA vs LISTA speed-accuracy tradeoff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ista_vs_lista_speed_accuracy.png", dpi=160)
    plt.close()

    print("\nBenchmark complete. Saved performance graphs:")
    print(f"- {plots_dir / 'ista_objective_vs_iteration.png'}")
    print(f"- {plots_dir / 'lista_loss_vs_epoch.png'}")
    print(f"- {plots_dir / 'ista_vs_lista_accuracy_progression.png'}")
    print(f"- {plots_dir / 'ista_vs_lista_speed_accuracy.png'}")


def _normalize_dictionary_columns(dictionary: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(dictionary, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return dictionary / norms


def _generate_sparse_codes(
    num_samples: int,
    code_dim: int,
    sparsity_level: int,
    rng: np.random.Generator,
) -> np.ndarray:
    codes = np.zeros((num_samples, code_dim), dtype=np.float32)
    for index in range(num_samples):
        support = rng.choice(code_dim, size=sparsity_level, replace=False)
        codes[index, support] = rng.normal(0.0, 1.0, size=sparsity_level).astype(np.float32)
    return codes


def _sparse_coding_objective(
    observations: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    lambda_reg: float,
) -> float:
    reconstruction = codes @ dictionary.T
    residual = reconstruction - observations
    mse_term = 0.5 * np.mean(np.sum(residual**2, axis=1))
    l1_term = lambda_reg * np.mean(np.sum(np.abs(codes), axis=1))
    return float(mse_term + l1_term)


def _ista_sparse_coding_curves(
    observations: np.ndarray,
    dictionary: np.ndarray,
    lambda_reg: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = observations.shape[0]
    code_dim = dictionary.shape[1]
    codes = np.zeros((samples, code_dim), dtype=np.float32)

    gram = dictionary.T @ dictionary
    spectral_norm = float(np.linalg.norm(gram, ord=2))
    step_size = 1.0 / float(np.maximum(spectral_norm, 1e-8))

    objective_curve: list[float] = []
    elapsed_curve: list[float] = []
    start_time = time.perf_counter()

    for _ in range(max_iter):
        gradient = (codes @ gram) - (observations @ dictionary)
        codes = np.sign(codes - step_size * gradient) * np.maximum(
            np.abs(codes - step_size * gradient) - lambda_reg * step_size,
            0.0,
        )

        objective = _sparse_coding_objective(observations, dictionary, codes, lambda_reg)
        objective_curve.append(objective)
        elapsed_curve.append(time.perf_counter() - start_time)

    iteration_axis = np.arange(1, max_iter + 1, dtype=np.float32)
    return iteration_axis, np.asarray(objective_curve, dtype=np.float32), np.asarray(elapsed_curve, dtype=np.float32)


def _ista_sparse_codes(
    observations: np.ndarray,
    dictionary: np.ndarray,
    lambda_reg: float,
    max_iter: int,
) -> np.ndarray:
    samples = observations.shape[0]
    code_dim = dictionary.shape[1]
    codes = np.zeros((samples, code_dim), dtype=np.float32)

    gram = dictionary.T @ dictionary
    spectral_norm = float(np.linalg.norm(gram, ord=2))
    step_size = 1.0 / float(np.maximum(spectral_norm, 1e-8))

    for _ in range(max_iter):
        gradient = (codes @ gram) - (observations @ dictionary)
        codes = np.sign(codes - step_size * gradient) * np.maximum(
            np.abs(codes - step_size * gradient) - lambda_reg * step_size,
            0.0,
        )

    return codes


def _train_lista_sparse_coder(
    observations_train: np.ndarray,
    target_codes_train: np.ndarray,
    dictionary: np.ndarray,
    lambda_reg: float,
    input_dim: int,
    code_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> LISTASparseCoder:
    model = LISTASparseCoder(input_dim=input_dim, code_dim=code_dim, num_layers=num_layers).to(device)
    model.initialize_from_ista(dictionary=dictionary, lambda_reg=lambda_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(
        torch.from_numpy(observations_train).float(),
        torch.from_numpy(target_codes_train).float(),
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch_obs, batch_codes in loader:
            batch_obs = batch_obs.to(device)
            batch_codes = batch_codes.to(device)

            optimizer.zero_grad()
            code_outputs = model(batch_obs, return_all=True)
            if not isinstance(code_outputs, list) or len(code_outputs) == 0:
                raise RuntimeError("LISTA forward did not return layer outputs.")

            layer_losses = [nn.functional.mse_loss(output, batch_codes) for output in code_outputs]
            loss = torch.stack(layer_losses).mean()
            loss.backward()
            optimizer.step()

    return model


def _lista_sparse_coding_curves(
    model: LISTASparseCoder,
    observations: np.ndarray,
    dictionary: np.ndarray,
    lambda_reg: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    observations_tensor = torch.from_numpy(observations).float().to(device)

    with torch.no_grad():
        outputs = model(observations_tensor, return_all=True)
    if not isinstance(outputs, list) or len(outputs) == 0:
        raise RuntimeError("LISTA forward did not return any layer outputs.")

    objective_curve: list[float] = []
    elapsed_curve: list[float] = []

    running_start = time.perf_counter()
    with torch.no_grad():
        y_term = observations_tensor @ model.w_e.T
        code = torch.zeros(
            observations_tensor.shape[0],
            model.w_e.shape[0],
            dtype=observations_tensor.dtype,
            device=observations_tensor.device,
        )
        for layer_idx in range(model.num_layers):
            code = model._soft_threshold(
                y_term + code @ model.s_layers[layer_idx].T,
                model.theta[layer_idx].unsqueeze(0),
            )
            code_np = code.detach().cpu().numpy().astype(np.float32)
            objective_curve.append(
                _sparse_coding_objective(observations, dictionary, code_np, lambda_reg)
            )
            elapsed_curve.append(time.perf_counter() - running_start)

    layer_axis = np.arange(1, model.num_layers + 1, dtype=np.float32)
    return layer_axis, np.asarray(objective_curve, dtype=np.float32), np.asarray(elapsed_curve, dtype=np.float32)


