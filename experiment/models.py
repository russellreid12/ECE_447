from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


class ISTAClassifier:
    """Linear multi-class classifier trained with ISTA-style proximal updates.

    Objective optimized in fit:
        0.5 * ||XW + b - Y||_2^2 + lambda * ||W||_1

    where Y is a one-hot encoding of class labels. The L1 term drives sparse
    weights, and the proximal (soft-threshold) step handles that non-smooth term.
    """

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        max_iter: int = 300,
        tol: float = 1e-5,
        verbose: bool = True,
    ) -> None:
        # lambda_reg controls sparsity strength; higher values yield more zeros in W.
        self.lambda_reg = lambda_reg
        # max_iter is the maximum number of proximal-gradient iterations.
        self.max_iter = max_iter
        # tol is the relative parameter-change threshold for early stopping.
        self.tol = tol
        self.verbose = verbose
        # Learned linear classifier parameters (set after fit).
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        # Per-iteration diagnostics (objective, mse, l1, relative change, time).
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        """Proximal operator for L1 norm: sign(v) * max(|v|-t, 0)."""
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def fit(self, features: np.ndarray, labels: np.ndarray, num_classes: int) -> None:
        """Train the sparse linear classifier with ISTA updates."""
        samples, num_features = features.shape
        # Convert class IDs to one-hot vectors for squared-loss training.
        one_hot = np.eye(num_classes)[labels]
        start_time = time.perf_counter()

        # Initialize weights and bias at zero.
        weights = np.zeros((num_features, num_classes), dtype=np.float32)
        bias = np.zeros(num_classes, dtype=np.float32)

        # Estimate Lipschitz constant of gradient and set a stable step size.
        spectral_norm = np.linalg.norm(features, ord=2)
        lipschitz = max((spectral_norm**2) / samples, 1e-8)
        step_size = 1.0 / lipschitz

        for iteration in range(1, self.max_iter + 1):
            # Current predictions and residual in one-hot space.
            logits = features @ weights + bias
            residual = logits - one_hot

            # Gradient of smooth loss term with respect to W and b.
            grad_weights = (features.T @ residual) / samples
            grad_bias = residual.mean(axis=0)

            # Gradient step on W followed by soft-thresholding for L1 sparsity.
            next_weights = self._soft_threshold(
                weights - step_size * grad_weights,
                self.lambda_reg * step_size,
            )
            # Bias has no L1 penalty, so plain gradient step.
            next_bias = bias - step_size * grad_bias

            # Relative parameter change used for convergence detection.
            rel_change = np.linalg.norm(next_weights - weights) / (np.linalg.norm(weights) + 1e-8)

            weights = next_weights
            bias = next_bias

            # Track objective components for plotting/debugging.
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

            # Stop early when updates are sufficiently small.
            if rel_change < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration} (rel_change={rel_change:.6e}).")
                break

        # Persist trained parameters.
        self.weights = weights
        self.bias = bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class IDs via argmax over linear logits."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        logits = features @ self.weights + self.bias
        return np.argmax(logits, axis=1)


class LISTAClassifier(nn.Module):
    """Unrolled LISTA network for classification.

    The model performs a fixed number of learned sparse-coding steps (num_layers)
    and then classifies using the final code vector.
    """

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        num_classes: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        # Number of unrolled ISTA-like layers.
        self.num_layers = num_layers
        # Learned input projection term (analog of W_e in LISTA literature).
        self.w_x = nn.Parameter(torch.empty(code_dim, input_dim))
        # Learned recurrent matrices, one per unrolled layer.
        self.s_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(code_dim, code_dim)) for _ in range(num_layers)]
        )
        # Layer-wise shrinkage thresholds.
        self.theta = nn.Parameter(torch.ones(num_layers, code_dim) * 0.1)
        # Classifier head from sparse code to logits.
        self.classifier = nn.Linear(code_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Xavier initialization keeps activations in a reasonable range at start.
        nn.init.xavier_uniform_(self.w_x)
        for matrix in self.s_matrices:
            nn.init.xavier_uniform_(matrix)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Element-wise soft-thresholding used to induce sparse codes."""
        return torch.sign(values) * torch.relu(torch.abs(values) - threshold)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run unrolled sparse-coding layers, then classify with final code."""
        batch_size = features.shape[0]
        code_dim = self.w_x.shape[0]
        # Input contribution reused at every layer.
        x_term = features @ self.w_x.T

        # Start from zero code and refine it through learned unrolled updates.
        code = torch.zeros(batch_size, code_dim, device=features.device, dtype=features.dtype)
        for layer in range(self.num_layers):
            # LISTA update: z_{k+1} = S_theta(x_term + S_k z_k)
            recurrent_term = code @ self.s_matrices[layer].T
            pre_activation = x_term + recurrent_term
            code = self._soft_threshold(pre_activation, self.theta[layer].unsqueeze(0))

        # Use final sparse code to produce class logits.
        logits = self.classifier(code)
        return logits, code


class LISTASparseCoder(nn.Module):
    """Unrolled LISTA network for sparse coding (no classifier head)."""

    def __init__(self, input_dim: int, code_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        # Input projection term for sparse coding.
        self.w_e = nn.Parameter(torch.empty(code_dim, input_dim))
        # Recurrent LISTA matrices per layer.
        self.s_layers = nn.ParameterList(
            [nn.Parameter(torch.empty(code_dim, code_dim)) for _ in range(num_layers)]
        )
        # Layer-wise shrinkage thresholds.
        self.theta = nn.Parameter(torch.ones(num_layers, code_dim) * 0.1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Standard neural init for learnable matrices.
        nn.init.xavier_uniform_(self.w_e)
        for matrix in self.s_layers:
            nn.init.xavier_uniform_(matrix)

    def initialize_from_ista(self, dictionary: np.ndarray, lambda_reg: float) -> None:
        """Initialize LISTA parameters from classical ISTA operators.

        This warm start typically speeds convergence and aligns the model with
        the underlying sparse coding objective before gradient training.
        """
        # Gram matrix and its spectral norm define ISTA step scaling.
        gram = dictionary.T @ dictionary
        spectral_norm = float(np.linalg.norm(gram, ord=2))
        lipschitz = float(np.maximum(spectral_norm, 1e-8))

        # ISTA-equivalent initial parameters.
        w_e_init = (dictionary.T / lipschitz).astype(np.float32)
        s_init = (np.eye(dictionary.shape[1], dtype=np.float32) - (gram / lipschitz)).astype(np.float32)
        theta_init = np.full((self.num_layers, dictionary.shape[1]), lambda_reg / lipschitz, dtype=np.float32)

        # Copy NumPy initial values into torch parameters without tracking gradients.
        with torch.no_grad():
            self.w_e.copy_(torch.from_numpy(w_e_init).to(self.w_e.device))
            for matrix in self.s_layers:
                matrix.copy_(torch.from_numpy(s_init).to(matrix.device))
            self.theta.copy_(torch.from_numpy(theta_init).to(self.theta.device))

    @staticmethod
    def _soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Element-wise soft-thresholding used in each unrolled layer."""
        return torch.sign(values) * torch.relu(torch.abs(values) - threshold)

    def forward(self, observations: torch.Tensor, return_all: bool = False) -> torch.Tensor | list[torch.Tensor]:
        """Compute sparse code after num_layers updates.

        If return_all=True, also return the intermediate code after each layer,
        which is useful for diagnostics and layer-wise losses.
        """
        batch_size = observations.shape[0]
        code_dim = self.w_e.shape[0]
        # Start from zero code and reuse the input-projection term each layer.
        code = torch.zeros(batch_size, code_dim, dtype=observations.dtype, device=observations.device)
        y_term = observations @ self.w_e.T
        outputs: list[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            # Unrolled LISTA sparse-coding update.
            code = self._soft_threshold(
                y_term + code @ self.s_layers[layer_idx].T,
                self.theta[layer_idx].unsqueeze(0),
            )
            if return_all:
                # Keep intermediate layers for analysis/training objectives.
                outputs.append(code)

        if return_all:
            return outputs
        return code
