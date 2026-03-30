from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2


def generate_mnist(output_dir: Path, raw_data_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_dataset = torchvision.datasets.MNIST(
        root=str(raw_data_dir),
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=str(raw_data_dir),
        train=False,
        transform=transform,
        download=True,
    )

    x_train = torch.stack([image for image, _ in train_dataset])
    y_train = torch.tensor([label for _, label in train_dataset], dtype=torch.long)

    x_test = torch.stack([image for image, _ in test_dataset])
    y_test = torch.tensor([label for _, label in test_dataset], dtype=torch.long)

    torch.save(x_train, output_dir / "x_train.pt")
    torch.save(y_train, output_dir / "y_train.pt")
    torch.save(x_test, output_dir / "x_test.pt")
    torch.save(y_test, output_dir / "y_test.pt")

    print(f"Saved MNIST tensors to: {output_dir}")
    print(f"x_train: {tuple(x_train.shape)} | y_train: {tuple(y_train.shape)}")
    print(f"x_test: {tuple(x_test.shape)} | y_test: {tuple(y_test.shape)}")


def load_mnist_tensors(mnist_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train_path = mnist_dir / "x_train.pt"
    y_train_path = mnist_dir / "y_train.pt"
    x_test_path = mnist_dir / "x_test.pt"
    y_test_path = mnist_dir / "y_test.pt"

    required_paths = [x_train_path, y_train_path, x_test_path, y_test_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Missing MNIST tensor files. Run with '--task generate' first. Missing:\n"
            f"{missing_text}"
        )

    x_train = torch.load(x_train_path)
    y_train = torch.load(y_train_path)
    x_test = torch.load(x_test_path)
    y_test = torch.load(y_test_path)
    return x_train, y_train, x_test, y_test


def get_preprocessed_mnist_arrays(
    mnist_dir: Path,
    max_train_samples: int,
    max_test_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return x_train, y_train, x_test, y_test
