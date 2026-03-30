from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


class ISTAClassifier:
    def __init__(
        self,
        lambda_reg: float = 1e-3,
        max_iter: int = 300,
        tol: float = 1e-5,
        verbose: bool = True,
    ) -> None:
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def fit(self, features: np.ndarray, labels: np.ndarray, num_classes: int) -> None:
        samples, num_features = features.shape
        one_hot = np.eye(num_classes)[labels]
        start_time = time.perf_counter()

        weights = np.zeros((num_features, num_classes), dtype=np.float32)
        bias = np.zeros(num_classes, dtype=np.float32)

        spectral_norm = np.linalg.norm(features, ord=2)
        lipschitz = max((spectral_norm**2) / samples, 1e-8)
        step_size = 1.0 / lipschitz

        for iteration in range(1, self.max_iter + 1):
            logits = features @ weights + bias
            residual = logits - one_hot

            grad_weights = (features.T @ residual) / samples
            grad_bias = residual.mean(axis=0)

            next_weights = self._soft_threshold(
                weights - step_size * grad_weights,
                self.lambda_reg * step_size,
            )
            next_bias = bias - step_size * grad_bias

            rel_change = np.linalg.norm(next_weights - weights) / (np.linalg.norm(weights) + 1e-8)

            weights = next_weights
            bias = next_bias

            mse_loss = 0.5 * np.mean((features @ weights + bias - one_hot) ** 2)
            l1_penalty = self.lambda_reg * np.sum(np.abs(weights))
            objective = mse_loss + l1_penalty
            elapsed_sec = time.perf_counter() - start_time
            self.history.append(
                {
                    "iteration": float(iteration),
                    "objective": float(objective),
                    "mse": float(mse_loss),
                    "l1": float(l1_penalty),
                    "rel_change": float(rel_change),
                    "elapsed_sec": float(elapsed_sec),
                }
            )

            if self.verbose and (iteration == 1 or iteration % 25 == 0 or iteration == self.max_iter):
                print(
                    f"Iter {iteration:4d} | objective={objective:.6f} "
                    f"| mse={mse_loss:.6f} | l1={l1_penalty:.6f} | rel_change={rel_change:.6e}"
                )

            if rel_change < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration} (rel_change={rel_change:.6e}).")
                break

        self.weights = weights
        self.bias = bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        logits = features @ self.weights + self.bias
        return np.argmax(logits, axis=1)


class LISTAClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        num_classes: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.w_x = nn.Parameter(torch.empty(code_dim, input_dim))
        self.s_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(code_dim, code_dim)) for _ in range(num_layers)]
        )
        self.theta = nn.Parameter(torch.ones(num_layers, code_dim) * 0.1)
        self.classifier = nn.Linear(code_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_x)
        for matrix in self.s_matrices:
            nn.init.xavier_uniform_(matrix)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return torch.sign(values) * torch.relu(torch.abs(values) - threshold)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        code_dim = self.w_x.shape[0]
        x_term = features @ self.w_x.T

        code = torch.zeros(batch_size, code_dim, device=features.device, dtype=features.dtype)
        for layer in range(self.num_layers):
            recurrent_term = code @ self.s_matrices[layer].T
            pre_activation = x_term + recurrent_term
            code = self._soft_threshold(pre_activation, self.theta[layer].unsqueeze(0))

        logits = self.classifier(code)
        return logits, code


class LISTASparseCoder(nn.Module):
    def __init__(self, input_dim: int, code_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.w_e = nn.Parameter(torch.empty(code_dim, input_dim))
        self.s_layers = nn.ParameterList(
            [nn.Parameter(torch.empty(code_dim, code_dim)) for _ in range(num_layers)]
        )
        self.theta = nn.Parameter(torch.ones(num_layers, code_dim) * 0.1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_e)
        for matrix in self.s_layers:
            nn.init.xavier_uniform_(matrix)

    def initialize_from_ista(self, dictionary: np.ndarray, lambda_reg: float) -> None:
        gram = dictionary.T @ dictionary
        spectral_norm = float(np.linalg.norm(gram, ord=2))
        lipschitz = float(np.maximum(spectral_norm, 1e-8))

        w_e_init = (dictionary.T / lipschitz).astype(np.float32)
        s_init = (np.eye(dictionary.shape[1], dtype=np.float32) - (gram / lipschitz)).astype(np.float32)
        theta_init = np.full((self.num_layers, dictionary.shape[1]), lambda_reg / lipschitz, dtype=np.float32)

        with torch.no_grad():
            self.w_e.copy_(torch.from_numpy(w_e_init).to(self.w_e.device))
            for matrix in self.s_layers:
                matrix.copy_(torch.from_numpy(s_init).to(matrix.device))
            self.theta.copy_(torch.from_numpy(theta_init).to(self.theta.device))

    @staticmethod
    def _soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return torch.sign(values) * torch.relu(torch.abs(values) - threshold)

    def forward(self, observations: torch.Tensor, return_all: bool = False) -> torch.Tensor | list[torch.Tensor]:
        batch_size = observations.shape[0]
        code_dim = self.w_e.shape[0]
        code = torch.zeros(batch_size, code_dim, dtype=observations.dtype, device=observations.device)
        y_term = observations @ self.w_e.T
        outputs: list[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            code = self._soft_threshold(
                y_term + code @ self.s_layers[layer_idx].T,
                self.theta[layer_idx].unsqueeze(0),
            )
            if return_all:
                outputs.append(code)

        if return_all:
            return outputs
        return code
