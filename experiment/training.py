from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_mnist_tensors
from .models import ISTAClassifier, LISTAClassifier


def train_ista(
    mnist_dir: Path,
    max_train_samples: int,
    max_test_samples: int,
    lambda_reg: float,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> dict[str, object] | None:
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = load_mnist_tensors(mnist_dir)

    if max_train_samples > 0:
        x_train_tensor = x_train_tensor[:max_train_samples]
        y_train_tensor = y_train_tensor[:max_train_samples]
    if max_test_samples > 0:
        x_test_tensor = x_test_tensor[:max_test_samples]
        y_test_tensor = y_test_tensor[:max_test_samples]

    x_train = x_train_tensor.reshape(x_train_tensor.shape[0], -1).numpy().astype(np.float32)
    y_train = y_train_tensor.numpy().astype(np.int64)
    x_test = x_test_tensor.reshape(x_test_tensor.shape[0], -1).numpy().astype(np.float32)
    y_test = y_test_tensor.numpy().astype(np.int64)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = ISTAClassifier(lambda_reg=lambda_reg, max_iter=max_iter, tol=tol, verbose=verbose)
    model.fit(x_train, y_train, num_classes=10)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    if model.weights is None:
        raise RuntimeError("Model weights are unavailable after training.")

    non_zero = int(np.count_nonzero(model.weights))
    total = int(model.weights.size)
    sparsity = 1.0 - (non_zero / total)

    print("\nISTA training complete.")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")
    print(f"Weight sparsity: {sparsity:.4f} ({non_zero}/{total} non-zero)")

    return {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "sparsity": float(sparsity),
        "history": model.history,
    }


def train_lista(
    mnist_dir: Path,
    max_train_samples: int,
    max_test_samples: int,
    lambda_reg: float,
    lista_layers: int,
    lista_code_dim: int,
    lista_epochs: int,
    lista_batch_size: int,
    lista_lr: float,
    lista_checkpoint_path: Path,
    load_lista_checkpoint: bool,
    save_lista_checkpoint: bool,
    verbose: bool,
) -> dict[str, object] | None:
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = load_mnist_tensors(mnist_dir)

    if max_train_samples > 0:
        x_train_tensor = x_train_tensor[:max_train_samples]
        y_train_tensor = y_train_tensor[:max_train_samples]
    if max_test_samples > 0:
        x_test_tensor = x_test_tensor[:max_test_samples]
        y_test_tensor = y_test_tensor[:max_test_samples]

    x_train_np = x_train_tensor.reshape(x_train_tensor.shape[0], -1).numpy().astype(np.float32)
    y_train_np = y_train_tensor.numpy().astype(np.int64)
    x_test_np = x_test_tensor.reshape(x_test_tensor.shape[0], -1).numpy().astype(np.float32)
    y_test_np = y_test_tensor.numpy().astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_lista_checkpoint:
        if not lista_checkpoint_path.exists():
            raise FileNotFoundError(
                f"LISTA checkpoint not found: {lista_checkpoint_path}. "
                "Train once with '--save-lista-checkpoint' first."
            )

        checkpoint = torch.load(lista_checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_input_dim = int(checkpoint["input_dim"])
        checkpoint_code_dim = int(checkpoint["code_dim"])
        checkpoint_layers = int(checkpoint["num_layers"])

        if checkpoint_input_dim != x_train_np.shape[1]:
            raise ValueError(
                "Checkpoint input dimension does not match current data. "
                f"checkpoint={checkpoint_input_dim}, data={x_train_np.shape[1]}"
            )

        model = LISTAClassifier(
            input_dim=checkpoint_input_dim,
            code_dim=checkpoint_code_dim,
            num_classes=10,
            num_layers=checkpoint_layers,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        scaler_mean = checkpoint["scaler_mean"]
        scaler_scale = checkpoint["scaler_scale"]
        if isinstance(scaler_mean, torch.Tensor):
            scaler_mean = scaler_mean.cpu().numpy()
        if isinstance(scaler_scale, torch.Tensor):
            scaler_scale = scaler_scale.cpu().numpy()

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaler_mean, dtype=np.float32)
        scaler.scale_ = np.asarray(scaler_scale, dtype=np.float32)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = scaler.mean_.shape[0]

        if verbose:
            print(f"Loaded LISTA checkpoint: {lista_checkpoint_path}")
            print(
                "Using checkpoint architecture: "
                f"layers={checkpoint_layers}, code_dim={checkpoint_code_dim}"
            )
    else:
        scaler = StandardScaler()
        x_train_np = scaler.fit_transform(x_train_np)
        model = LISTAClassifier(
            input_dim=x_train_np.shape[1],
            code_dim=lista_code_dim,
            num_classes=10,
            num_layers=lista_layers,
        ).to(device)

    x_test_np = scaler.transform(x_test_np)
    if load_lista_checkpoint:
        x_train_np = scaler.transform(x_train_np)

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_test = torch.from_numpy(x_test_np)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=lista_batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lista_lr)

    start_time = time.perf_counter()
    history: list[dict[str, float]] = []

    if lista_epochs > 0:
        for epoch in range(1, lista_epochs + 1):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for features_batch, labels_batch in train_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)

                optimizer.zero_grad()
                logits, sparse_code = model(features_batch)
                classification_loss = nn.functional.cross_entropy(logits, labels_batch)
                sparse_penalty = lambda_reg * torch.mean(torch.abs(sparse_code))
                loss = classification_loss + sparse_penalty
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * labels_batch.shape[0]
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.shape[0]

            avg_loss = epoch_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            model.eval()
            with torch.no_grad():
                test_logits_epoch, _ = model(x_test.to(device))
                test_predictions_epoch = torch.argmax(test_logits_epoch, dim=1).cpu().numpy()
                test_acc = accuracy_score(y_test_np, test_predictions_epoch)

            elapsed_sec = time.perf_counter() - start_time
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(avg_loss),
                    "train_accuracy": float(train_acc),
                    "test_accuracy": float(test_acc),
                    "elapsed_sec": float(elapsed_sec),
                }
            )

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{lista_epochs} | train_loss={avg_loss:.6f} "
                    f"| train_acc={train_acc:.4f}"
                )
    elif verbose:
        print("Skipping LISTA training because --lista-epochs is <= 0.")

    model.eval()
    with torch.no_grad():
        train_logits, train_codes = model(x_train.to(device))
        test_logits, _ = model(x_test.to(device))

        train_predictions = torch.argmax(train_logits, dim=1).cpu().numpy()
        test_predictions = torch.argmax(test_logits, dim=1).cpu().numpy()

        train_accuracy = accuracy_score(y_train_np, train_predictions)
        test_accuracy = accuracy_score(y_test_np, test_predictions)

        mean_abs_code = torch.mean(torch.abs(train_codes)).item()
        near_zero_fraction = (torch.abs(train_codes) < 1e-3).float().mean().item()
        train_codes_non_zero = int((torch.abs(train_codes) >= 1e-3).sum().item())
        train_codes_total = int(train_codes.numel())

    print("\nLISTA training complete.")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")
    print(f"Mean |code|: {mean_abs_code:.6f}")
    print(
        "Code sparsity (|code| < 1e-3): "
        f"{near_zero_fraction:.4f} ({train_codes_non_zero}/{train_codes_total} active)"
    )

    if save_lista_checkpoint:
        lista_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": int(x_train.shape[1]),
                "code_dim": int(model.w_x.shape[0]),
                "num_layers": int(model.num_layers),
                "lambda_reg": float(lambda_reg),
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
            },
            lista_checkpoint_path,
        )
        print(f"Saved LISTA checkpoint: {lista_checkpoint_path}")

    return {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "mean_abs_code": float(mean_abs_code),
        "code_sparsity": float(near_zero_fraction),
        "history": history,
    }
