"""
ECE447 Project 1 — LISTA Reproduction
Gregor & LeCun (2010): "Learning Fast Approximations of Sparse Coding"

Reproduces:
  - Figure 3: Code prediction error vs. iterations (FISTA vs LISTA)
  - Table 1: Error at T=1,3,7 iterations for different dictionary sizes

This mirrors the paper's exact setup:
  - Dictionary learning uses CoD (not ISTA) to generate Z* at each step
  - Z* targets for LISTA training are generated using CoD to convergence
  - The evaluation baseline is FISTA (not ISTA) — this is what Figure 3 shows
  - ISTA does not appear anywhere in this pipeline

MULTIPLE TRIALS NOTE:
  We train each LISTA configuration (depth T, dict size m) with NUM_SEEDS
  different random seeds. This gives error bars on our plots and satisfies
  the rubric's "multiple runs when necessary" requirement. We do NOT need
  to re-run FISTA multiple times — it is deterministic given a fixed dictionary.
  The variance comes from LISTA's random weight initialization and SGD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import csv

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
# These match the paper's two main conditions: m=100 (complete) and m=400 (4x overcomplete).
# n=100 because MNIST patches are 10x10.
# WHY patches instead of full images? The paper explicitly uses 10x10 image patches.
# This keeps n=100, which makes the math tractable and matches the paper's Table 1.

config = {
    # Data
    "patch_size": 10,          # 10x10 patches → n=100 input dim
    "n_patches_train": 50000,
    "n_patches_test": 5000,

    # Dictionary
    "dict_sizes": [100, 400],  # m: number of atoms. 100=complete, 400=overcomplete
    "dict_lr": 1e-3,
    "dict_epochs": 5,

    # Sparse coding
    "alpha": 0.5,              # sparsity penalty weight (matches paper exactly)
    "cod_max_iters": 500,      # CoD iterations to get converged Z* targets
    "cod_tol": 1e-6,           # convergence threshold

    # LISTA training
    "depths": [1, 3, 7],       # T: number of unrolled steps (matches paper's Table 1)
    "lista_epochs": 10,
    "lista_lr": 1e-3,
    "batch_size": 256,

    # Experiment
    "num_seeds": 3,            # random seeds per (T, m) condition → gives error bars

    # Device
    "device": (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    ),
}

print(f"Using device: {config['device']}")


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_mnist_patches(n_train, n_test, patch_size):
    """
    Load MNIST and extract random 10x10 patches.
    Each patch is flattened to a vector of length patch_size^2 = 100.
    Preprocessing: zero-mean, unit-variance (matches paper's preprocessing).
    """
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    dataset = torchvision.datasets.MNIST("data", train=True, transform=transform, download=True)

    all_images = torch.stack([dataset[i][0].squeeze() for i in range(len(dataset))])

    def extract_patches(images, n_patches):
        H, W = images.shape[1], images.shape[2]
        p = patch_size
        patches = []
        for _ in range(n_patches):
            idx = torch.randint(0, len(images), (1,)).item()
            r = torch.randint(0, H - p + 1, (1,)).item()
            c = torch.randint(0, W - p + 1, (1,)).item()
            patch = images[idx, r:r+p, c:c+p].flatten()
            patches.append(patch)
        patches = torch.stack(patches)

        # Normalize: subtract mean, divide by std
        # WHY: The paper discards patches with small std. We clamp to avoid division by zero.
        means = patches.mean(dim=1, keepdim=True)
        stds = patches.std(dim=1, keepdim=True).clamp(min=1e-6)
        patches = (patches - means) / stds
        return patches

    print("Extracting training patches...")
    X_train = extract_patches(all_images, n_train)
    print("Extracting test patches...")
    X_test = extract_patches(all_images, n_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test


# =============================================================================
# 2. SHARED UTILITY — soft thresholding
# =============================================================================

def soft_threshold(x, theta):
    """
    Shrinkage function h_theta(x) = sign(x) * max(|x| - theta, 0).
    This is the proximal operator for the L1 penalty.
    Used in CoD, FISTA, and LISTA.
    """
    return torch.sign(x) * F.relu(x.abs() - theta)


# =============================================================================
# 3. COORDINATE DESCENT (CoD) — Algorithm 2 from the paper
# =============================================================================
# WHY CoD and not ISTA? The paper explicitly uses CoD as its preferred exact
# solver for both dictionary learning and Z* target generation. CoD updates
# one carefully-chosen coordinate at a time (O(m) per step) rather than all
# coordinates simultaneously (O(m^2) for ISTA), making it faster per iteration.
#
# This is used in two places:
#   - Inside learn_dictionary: rough Z* to guide W_d gradient steps
#   - Inside generate_targets: precise Z* as LISTA's training labels

def cod(X, Wd, alpha, max_iters, tol=1e-6):
    """
    Coordinate Descent sparse coding — Algorithm 2 from Gregor & LeCun (2010).

    At each step, picks the code component that would change most and updates
    only that one. Each step costs O(m) vs ISTA's O(m^2).

    Args:
        X:         input patches, shape (batch, n)
        Wd:        dictionary, shape (n, m)
        alpha:     sparsity penalty
        max_iters: maximum iterations
        tol:       stop if total change in Z drops below this

    Returns:
        Z: sparse code, shape (batch, m)
    """
    batched = X.dim() == 2
    if not batched:
        X = X.unsqueeze(0)

    n, m = Wd.shape
    batch = X.shape[0]
    device = X.device

    # S = I - Wd^T @ Wd (mutual inhibition matrix, as in Algorithm 2)
    # WHY: S propagates the effect of updating one code component to all others.
    # When component k changes by e, every Bj shifts by S_jk * e.
    WtW = Wd.T @ Wd                              # (m, m)
    S = torch.eye(m, device=device) - WtW        # (m, m)

    # Initialize: B = Wd^T @ X, Z = 0
    # B accumulates the "effective input" after accounting for already-active components
    B = X @ Wd                                   # (batch, m) — equivalent to Wd^T @ X per sample
    Z = torch.zeros(batch, m, device=device)

    for t in range(max_iters):
        Z_prev = Z.clone()

        # Candidate update for all components
        Z_bar = soft_threshold(B, alpha)          # (batch, m)

        # Find the component with the largest change for each sample in the batch
        diff = (Z_bar - Z).abs()                  # (batch, m)
        k = diff.argmax(dim=1)                    # (batch,) — one index per sample

        # Update only the chosen component for each sample
        # WHY a Python loop here? CoD is inherently sequential — each step
        # depends on the previous one. Vectorizing across the batch dimension
        # is possible but adds complexity. For target generation (run once,
        # cached), this is fast enough.
        for b in range(batch):
            kb = k[b].item()
            e = Z_bar[b, kb] - Z[b, kb]          # scalar change for this component
            B[b] += S[:, kb] * e                  # propagate change to all B entries
            Z[b, kb] = Z_bar[b, kb]               # update only this component

        # Convergence check
        if (Z - Z_prev).norm() < tol:
            break

    # Final shrinkage pass
    Z = soft_threshold(B, alpha)

    if not batched:
        Z = Z.squeeze(0)
    return Z


# =============================================================================
# 4. DICTIONARY LEARNING
# =============================================================================
# The paper's exact procedure (Section 4):
#   (1) get image patch X_p
#   (2) compute Z* using CoD
#   (3) update W_d with one SGD step: W_d <- W_d - eta * dE/dW_d
#   (4) renormalize columns of W_d to unit norm
#   (5) repeat with 1/t decaying step size
#
# We use Adam instead of vanilla SGD with 1/t schedule — Adam adapts
# its step size automatically and is more stable in practice.
# We use CoD (matching the paper) for the Z* inference step.

def learn_dictionary(X_train, m, n, config):
    """
    Learn dictionary W_d by alternating CoD inference and SGD on W_d.
    """
    device = config["device"]
    print(f"\nLearning dictionary (m={m})...")

    # Initialize W_d randomly with unit-norm columns
    Wd = torch.randn(n, m, device=device)
    Wd = F.normalize(Wd, dim=0)

    Wd.requires_grad_(True)
    optimizer = torch.optim.Adam([Wd], lr=config["dict_lr"])

    loader = DataLoader(X_train, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(config["dict_epochs"]):
        total_loss = 0
        for X_batch in tqdm(loader, desc=f"  Dict epoch {epoch+1}/{config['dict_epochs']}"):
            X_batch = X_batch.to(device)

            # Step 1: get sparse codes via CoD (no grad — not differentiating through CoD)
            # WHY detach? We only want to update W_d via the reconstruction loss below,
            # not through the CoD computation itself.
            with torch.no_grad():
                Z_star = cod(X_batch, Wd.detach(), config["alpha"],
                             max_iters=100, tol=1e-4)

            # Step 2: reconstruction loss — dE/dW_d = -(X - W_d @ Z*) @ Z*^T
            # The L1 term drops out because it has no dependence on W_d.
            X_recon = Z_star @ Wd.T              # (batch, n)
            loss = F.mse_loss(X_recon, X_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step 3: renormalize columns — no atom should dominate by being large
            with torch.no_grad():
                Wd.data = F.normalize(Wd.data, dim=0)

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}: recon loss = {total_loss/len(loader):.4f}")

    return Wd.detach()


# =============================================================================
# 5. GENERATE Z* TARGETS via CoD
# =============================================================================
# WHY a separate stage? We run CoD to convergence (500 iters) on every
# training sample once and cache the results. If we did this inside the
# LISTA training loop, we'd rerun CoD 500 times per sample per epoch —
# prohibitively slow. Caching means we pay this cost once.
#
# WHY 500 iterations? These Z* values are LISTA's ground truth labels.
# They must be as close to truly converged as possible — noisy targets
# mean LISTA learns to approximate a noisy signal.

def generate_targets(X, Wd, config, desc="Generating Z* targets"):
    """Run CoD to convergence on all of X and cache the sparse codes Z*."""
    device = config["device"]
    Wd = Wd.to(device)
    all_Z = []

    loader = DataLoader(X, batch_size=128, shuffle=False)
    for X_batch in tqdm(loader, desc=desc):
        X_batch = X_batch.to(device)
        with torch.no_grad():
            Z = cod(X_batch, Wd, config["alpha"],
                    max_iters=config["cod_max_iters"],
                    tol=config["cod_tol"])
        all_Z.append(Z.cpu())

    return torch.cat(all_Z, dim=0)


# =============================================================================
# 6. THE LISTA MODEL
# =============================================================================
# WHY a custom nn.Module instead of nn.Linear layers?
# Because of weight tying: We and S are shared across all T steps.
# Standard nn.Sequential creates separate weight matrices per layer.
# We need to explicitly reuse the same matrices in a loop.
#
# Architecture:
#   Input: X of shape (batch, n)
#   Step 0: Z = h_theta(We @ X)            ← initial estimate
#   Step t: Z = h_theta(We @ X + S @ Z)    ← refined estimate
#   Output: Z of shape (batch, m)
#
# Parameters learned (shared across all T steps — weight tying):
#   We:    (m, n) — replaces (1/L)*Wd^T from ISTA analytically
#   S:     (m, m) — replaces I - (1/L)*Wd^T*Wd from ISTA analytically
#   theta: (m,)   — per-dimension threshold, replaces scalar alpha/L

class LISTA(nn.Module):
    def __init__(self, n, m, T, Wd=None):
        """
        Args:
            n:  input dimension (patch_size^2 = 100)
            m:  code dimension (dictionary size)
            T:  number of unrolled steps
            Wd: optional pre-trained dictionary for smart initialization
        """
        super().__init__()
        self.T = T
        self.n = n
        self.m = m

        self.We = nn.Parameter(torch.empty(m, n))
        self.S = nn.Parameter(torch.empty(m, m))
        self.theta = nn.Parameter(torch.ones(m) * 0.1)

        # Smart initialization from W_d
        # WHY: Starting from the ISTA-derived values gives LISTA a head start —
        # we know these are already "pretty good". Learning then refines them.
        if Wd is not None:
            with torch.no_grad():
                WtW = Wd.T @ Wd
                L = torch.linalg.eigvalsh(WtW).max().item()
                L = max(L, 1e-6)
                self.We.data = (Wd.T / L).clone()
                self.S.data = (torch.eye(m) - WtW / L).clone()
                self.theta.data = torch.full((m,), 0.5 / L)
        else:
            nn.init.xavier_uniform_(self.We)
            nn.init.xavier_uniform_(self.S)

    def forward(self, X, return_all_iters=False):
        """
        Forward pass through T unrolled steps.

        Args:
            X:                input, shape (batch, n)
            return_all_iters: if True, return Z at every step 0..T (for Figure 3)

        Returns:
            Z:      sparse code at step T, shape (batch, m)
            all_Z:  list of Z at steps 0..T (only if return_all_iters=True)
        """
        B = F.linear(X, self.We)           # (batch, m) — We @ X, shared across steps
        Z = soft_threshold(B, self.theta)  # step 0: initial estimate

        all_Z = [Z] if return_all_iters else None

        for t in range(self.T):
            C = B + F.linear(Z, self.S)    # We @ X + S @ Z
            Z = soft_threshold(C, self.theta)
            if return_all_iters:
                all_Z.append(Z)

        if return_all_iters:
            return Z, all_Z
        return Z


# =============================================================================
# 7. LISTA TRAINING
# =============================================================================
# LISTA is trained to minimize ||Z_predicted - Z*||^2, where Z* comes from CoD.
# This is supervised learning with explicit targets — much easier to optimize
# than the full sparse coding energy directly.
#
# Backprop through LISTA is like BPTT in RNNs:
#   dL/dS = sum_{t=1}^{T} dL/dZ(t) * dZ(t)/dS
# Since We, S, theta are shared across all T steps, their gradients accumulate
# across all steps. PyTorch handles this automatically via autograd.

def train_lista(X_train, Z_train, X_test, Z_test, Wd, m, T, seed, config):
    """Train one LISTA model for a given (m, T, seed) combination."""
    torch.manual_seed(seed)
    device = config["device"]

    n = X_train.shape[1]
    model = LISTA(n, m, T, Wd=Wd.to(device)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lista_lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    dataset = torch.utils.data.TensorDataset(X_train, Z_train)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    train_history = []

    for epoch in range(config["lista_epochs"]):
        model.train()
        epoch_loss = 0

        for X_batch, Z_batch in loader:
            X_batch = X_batch.to(device)
            Z_batch = Z_batch.to(device)

            Z_pred = model(X_batch)
            loss = F.mse_loss(Z_pred, Z_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        train_history.append(avg_loss)

        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}/{config['lista_epochs']}: loss={avg_loss:.4f}")

    return model, train_history


# =============================================================================
# 8. FISTA — the evaluation baseline (Figure 3)
# =============================================================================
# WHY FISTA and not ISTA? Figure 3 in the paper compares LISTA against FISTA,
# not plain ISTA. FISTA is a faster variant of ISTA that adds a momentum term,
# making it converge more quickly. It is still much slower than LISTA — the
# paper shows LISTA needs roughly 20x fewer iterations to reach the same error.
#
# FISTA plays no role in learning W_d or generating Z* targets. It only appears
# here as the comparison baseline in the evaluation phase.

def evaluate_fista_curve(X_test, Z_test, Wd, config, max_eval_iters=35):
    """
    Run FISTA step by step on X_test, recording mse(Z_current, Z_test) at each step.
    This produces the gray baseline curve in Figure 3.

    FISTA update:
      Z_new = h_theta(We @ X + S @ Y)         ← shrinkage applied to momentum variable Y
      t_new = (1 + sqrt(1 + 4*t^2)) / 2       ← update momentum coefficient
      Y_new = Z_new + ((t-1)/t_new)*(Z_new - Z_old)  ← momentum step

    We and S are fixed analytically from W_d — not learned.
    """
    device = config["device"]
    Wd = Wd.to(device)
    X_test = X_test.to(device)
    Z_test = Z_test.to(device)

    n, m = Wd.shape

    # Compute fixed matrices analytically from W_d (same as ISTA)
    WtW = Wd.T @ Wd
    L = torch.linalg.eigvalsh(WtW).max().item()
    L = max(L, 1e-6)
    We_fixed = Wd.T / L                                      # (m, n)
    S_fixed = torch.eye(m, device=device) - WtW / L          # (m, m)
    theta = config["alpha"] / L                              # scalar threshold

    # Initialize Z and momentum variable Y
    Z = torch.zeros(X_test.shape[0], m, device=device)
    Y = Z.clone()
    t_k = 1.0

    iters = [0]
    errors = [F.mse_loss(Z, Z_test).item()]

    for t in range(1, max_eval_iters + 1):
        Z_prev = Z.clone()

        # FISTA step: shrinkage applied to momentum variable Y (not Z directly)
        Z = soft_threshold(X_test @ We_fixed.T + Y @ S_fixed.T, theta)

        # Momentum coefficient update
        t_k_next = (1 + (1 + 4 * t_k ** 2) ** 0.5) / 2
        momentum = (t_k - 1) / t_k_next

        # Momentum variable update: extrapolate beyond Z
        Y = Z + momentum * (Z - Z_prev)
        t_k = t_k_next

        iters.append(t)
        errors.append(F.mse_loss(Z, Z_test).item())

    return iters, errors


# =============================================================================
# 9. LISTA EVALUATION
# =============================================================================

def evaluate_lista_curve(model, X_test, Z_test, config):
    """
    Evaluate a trained LISTA model at each intermediate step 0..T.
    Returns iters [0, 1, ..., T] and corresponding mse errors.
    """
    device = config["device"]
    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)
        Z_test = Z_test.to(device)
        _, all_Z = model(X_test, return_all_iters=True)

    iters = list(range(len(all_Z)))
    errors = [F.mse_loss(Z.to(device), Z_test).item() for Z in all_Z]
    return iters, errors


# =============================================================================
# 10. MAIN EXPERIMENT LOOP
# =============================================================================
# For each dictionary size m in [100, 400]:
#   1. Learn W_d via CoD + SGD
#   2. Generate Z* targets for train and test via CoD to convergence
#   3. Evaluate FISTA curve (once per m — deterministic)
#   4. For each depth T in [1, 3, 7]:
#      For each seed in [0, 1, 2]:
#        - Train LISTA
#        - Evaluate error curve at steps 0..T

def run_experiments(config):
    X_train, X_test = load_mnist_patches(
        config["n_patches_train"], config["n_patches_test"], config["patch_size"]
    )
    n = config["patch_size"] ** 2  # 100

    results = {}

    for m in config["dict_sizes"]:
        print(f"\n{'='*60}")
        print(f"Dictionary size m={m}")
        print(f"{'='*60}")

        results[m] = {}

        # Stage 1: Learn W_d using CoD
        Wd = learn_dictionary(X_train, m, n, config)

        # Stage 2: Generate Z* targets using CoD to convergence
        Z_train = generate_targets(X_train, Wd, config, f"Z* train (m={m})")
        Z_test  = generate_targets(X_test,  Wd, config, f"Z* test  (m={m})")

        # Stage 3: FISTA evaluation curve (once per m, deterministic)
        print(f"\nEvaluating FISTA curve (m={m})...")
        fista_iters, fista_errors = evaluate_fista_curve(X_test, Z_test, Wd, config)
        results[m]["fista"] = (fista_iters, fista_errors)

        # Stage 4: Train and evaluate LISTA for each depth T and seed
        for T in config["depths"]:
            print(f"\n--- LISTA depth T={T}, m={m} ---")
            lista_curves = []

            for seed in range(config["num_seeds"]):
                print(f"  Seed {seed+1}/{config['num_seeds']}")
                model, history = train_lista(
                    X_train, Z_train, X_test, Z_test, Wd, m, T, seed, config
                )
                iters, errors = evaluate_lista_curve(model, X_test, Z_test, config)
                lista_curves.append((iters, errors))
                print(f"    Errors: { {i: f'{e:.3f}' for i,e in zip(iters, errors)} }")

            results[m][T] = lista_curves

    return results


# =============================================================================
# 11. PLOTTING — Figure 3 reproduction
# =============================================================================

def plot_figure3(results, config):
    """
    Reproduce Figure 3: code prediction error vs. iterations, log-log scale.
    FISTA is the gray baseline. LISTA curves for T=1,3,7 with error bars.
    """
    fig, axes = plt.subplots(1, len(config["dict_sizes"]),
                              figsize=(12, 5), sharey=True)

    colors = {1: "red", 3: "orange", 7: "blue"}

    for ax, m in zip(axes, config["dict_sizes"]):
        # FISTA baseline — single curve, no error bars (deterministic)
        fista_iters, fista_errors = results[m]["fista"]
        ax.plot(fista_iters[:20], fista_errors[:20],
                "x--", color="gray", label="FISTA", linewidth=1.5, markersize=6)

        # LISTA curves for each depth T
        for T in config["depths"]:
            curves = results[m][T]
            all_errors = np.array([errors for iters, errors in curves])
            mean_errors = all_errors.mean(axis=0)
            std_errors  = all_errors.std(axis=0)
            iters = curves[0][0]

            ax.errorbar(iters, mean_errors,
                        yerr=std_errors,
                        fmt="o-",
                        color=colors[T],
                        label=f"LISTA T={T}",
                        linewidth=1.5,
                        markersize=6,
                        capsize=4)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel("Code Prediction Error (MSE)", fontsize=12)
        ax.set_title(f"m={m} ({'complete' if m==100 else '4x overcomplete'})",
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 2, 3, 5, 7, 10, 20])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.suptitle("Figure 3 Reproduction: FISTA vs LISTA\n(error bars = std over 3 seeds)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("figure3_reproduction.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure3_reproduction.png")
    plt.show()


def plot_figure3_linear(results, config):
    """Same as Figure 3 but with linear axes — easier to read absolute differences."""
    fig, axes = plt.subplots(1, len(config["dict_sizes"]),
                              figsize=(12, 5), sharey=False)

    colors = {1: "red", 3: "orange", 7: "blue"}

    for ax, m in zip(axes, config["dict_sizes"]):
        fista_iters, fista_errors = results[m]["fista"]
        ax.plot(fista_iters[:20], fista_errors[:20],
                "x--", color="gray", label="FISTA", linewidth=1.5, markersize=6)

        for T in config["depths"]:
            curves = results[m][T]
            all_errors = np.array([errors for iters, errors in curves])
            mean_errors = all_errors.mean(axis=0)
            std_errors  = all_errors.std(axis=0)
            iters = curves[0][0]

            ax.errorbar(iters, mean_errors,
                        yerr=std_errors,
                        fmt="o-",
                        color=colors[T],
                        label=f"LISTA T={T}",
                        linewidth=1.5,
                        markersize=6,
                        capsize=4)

        ax.set_xlabel("Iterations", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Code Prediction Error (MSE)", fontsize=12)
        ax.set_title(f"m={m} ({'complete' if m==100 else '4x overcomplete'})",
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Figure 3 Reproduction (Linear Scale): FISTA vs LISTA\n(error bars = std over 3 seeds)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("figure3_reproduction_linear.png", dpi=150, bbox_inches="tight")
    print("Saved figure3_reproduction_linear.png")
    plt.show()


# =============================================================================
# 12. PAPER-STYLE FIGURE 3 — final points only, matching paper's exact style
# =============================================================================
# The paper's Figure 3 plots:
#   - FISTA as a continuous curve (crosses connected by dashed line)
#   - LISTA as single dots at each depth T=1,3,7 (final error only)
#   - Log-log scale
#
# This matches the paper's presentation exactly — no intermediate steps shown
# for LISTA, just the final error of each separately trained model.

def plot_figure3_paper_style(results, config):
    """
    Reproduce Figure 3 in the paper's exact style:
      - One plot per dictionary size (side by side)
      - FISTA shown as full curve with crosses
      - LISTA shown as single dots at final iteration T only
      - Log-log scale
      - No trails between LISTA points
    """
    fig, axes = plt.subplots(1, len(config["dict_sizes"]),
                              figsize=(12, 5), sharey=True)

    lista_colors = {1: "red", 3: "orange", 7: "blue"}

    for ax, m in zip(axes, config["dict_sizes"]):

        # FISTA — full curve, crosses connected by dashed line
        fista_iters, fista_errors = results[m]["fista"]
        ax.plot(fista_iters[1:20], fista_errors[1:20],
                "x--", color="gray", label="FISTA",
                linewidth=1.5, markersize=7, markeredgewidth=1.5)

        # LISTA — single dot at final step T only, mean ± std across seeds
        for T in config["depths"]:
            curves = results[m][T]

            # Take only the LAST error value from each seed (final step T)
            final_errors = np.array([errors[-1] for iters, errors in curves])
            mean_err = final_errors.mean()
            std_err  = final_errors.std()

            ax.errorbar([T], [mean_err],
                        yerr=[std_err],
                        fmt="o",
                        color=lista_colors[T],
                        label=f"LISTA T={T}",
                        markersize=8,
                        capsize=5,
                        markeredgewidth=1.5,
                        zorder=5)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel("Code Prediction Error", fontsize=12)
        ax.set_title(f"m={m} ({'complete' if m==100 else '4x overcomplete'})",
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 2, 3, 5, 7, 10, 20])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.suptitle("Figure 3 Reproduction: FISTA vs LISTA\n"
                 "LISTA dots = final error at depth T (mean ± std over 3 seeds)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("figure3_paper_style.png", dpi=150, bbox_inches="tight")
    print("Saved figure3_paper_style.png")
    plt.show()

def plot_figure3_combined(results, config):
    """
    Reproduce Figure 3 with all conditions on a single plot — matching the
    paper's original presentation where m=100 and m=400 are overlaid.
 
    Paper uses:
      - Red  = m=100 (1x complete)
      - Blue = m=400 (4x overcomplete)
      - x markers = FISTA, dots = LISTA
      - Single dots for LISTA at final T only
      - Log-log scale
    """
    fig, ax = plt.subplots(figsize=(8, 6))
 
    # Colors matching the paper: red=1x, blue=4x
    m_colors = {100: "red", 400: "blue"}
    m_labels = {100: "1x", 400: "4x"}
 
    for m in config["dict_sizes"]:
        color = m_colors[m]
        label = m_labels[m]
 
        # FISTA — full curve with crosses
        fista_iters, fista_errors = results[m]["fista"]
        ax.plot(fista_iters[1:20], fista_errors[1:20],
                "x--", color=color,
                label=f"FISTA ({label})",
                linewidth=1.5, markersize=7, markeredgewidth=1.5,
                alpha=0.7)
 
        # LISTA — single dot at final step T only
        for T in config["depths"]:
            curves = results[m][T]
            final_errors = np.array([errors[-1] for iters, errors in curves])
            mean_err = final_errors.mean()
            std_err  = final_errors.std()
 
            # Only add label for first T to avoid legend clutter
            lista_label = f"LISTA ({label})" if T == config["depths"][0] else "_nolegend_"
 
            ax.errorbar([T], [mean_err],
                        yerr=[std_err],
                        fmt="o",
                        color=color,
                        label=lista_label,
                        markersize=8,
                        capsize=4,
                        markeredgewidth=1.5,
                        zorder=5)
 
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Code Prediction Error", fontsize=12)
    ax.set_title("Figure 3 Reproduction: FISTA vs LISTA\n"
                 "Red = m=100 (complete), Blue = m=400 (4x overcomplete)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks([1, 2, 3, 5, 7, 10, 20])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
 
    plt.tight_layout()
    plt.savefig("figure3_combined.png", dpi=150, bbox_inches="tight")
    print("Saved figure3_combined.png")
    plt.show()


# =============================================================================
# 13. TABLE 1 REPRODUCTION
# =============================================================================

def print_table1(results, config):
    """
    Reproduce Table 1: code prediction error at T=1,3,7 for m=100 and m=400.
    Compares our results against the paper's reported values.
    """
    table_lines = []
    table_lines.append("=" * 55)
    table_lines.append("Table 1 Reproduction: Code Prediction Error")
    table_lines.append("=" * 55)
    table_lines.append(f"{'Method':<25} {'m=100':>10} {'m=400':>10}")
    table_lines.append("-" * 55)

    csv_data = []

    for T in config["depths"]:
        curves_100 = results[100][T]
        curves_400 = results[400][T]

        e100_mean = np.mean([c[1][-1] for c in curves_100])
        e100_std  = np.std( [c[1][-1] for c in curves_100])
        e400_mean = np.mean([c[1][-1] for c in curves_400])
        e400_std  = np.std( [c[1][-1] for c in curves_400])

        line = f"LISTA T={T:<18} {e100_mean:>7.2f}±{e100_std:.2f}  {e400_mean:>7.2f}±{e400_std:.2f}"
        table_lines.append(line)
        csv_data.append({
            "Method": f"LISTA T={T}",
            "m=100 mean": e100_mean, "m=100 std": e100_std,
            "m=400 mean": e400_mean, "m=400 std": e400_std,
        })

    table_lines.append("-" * 55)

    for T in config["depths"]:
        fista_iters_100, fista_errors_100 = results[100]["fista"]
        fista_iters_400, fista_errors_400 = results[400]["fista"]
        if T < len(fista_errors_100):
            e100 = fista_errors_100[T]
            e400 = fista_errors_400[T]
            line = f"FISTA @ iter {T:<13} {e100:>10.2f}  {e400:>10.2f}"
            table_lines.append(line)
            csv_data.append({
                "Method": f"FISTA @ iter {T}",
                "m=100 mean": e100, "m=100 std": None,
                "m=400 mean": e400, "m=400 std": None,
            })

    table_lines.append("=" * 55)
    table_lines.append("\nPaper's Table 1 reference values:")
    table_lines.append("  LISTA T=1: m=100 → 1.50,  m=400 → 2.45")
    table_lines.append("  LISTA T=3: m=100 → 0.98,  m=400 → 2.12")
    table_lines.append("  LISTA T=7: m=100 → 0.52,  m=400 → 1.62")
    table_lines.append("  FISTA T=1: m=100 → 21.0,  m=400 → 22.0")

    for line in table_lines:
        print(line)

    with open("table1_results.txt", "w") as f:
        f.write("\n".join(table_lines))
    print("\nSaved table1_results.txt")

    with open("table1_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "m=100 mean", "m=100 std", "m=400 mean", "m=400 std"])
        writer.writeheader()
        writer.writerows(csv_data)
    print("Saved table1_results.csv")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Saving results to: {script_dir}")

    results_file = "experiment_results.pkl"

    if os.path.exists(results_file):
        print(f"Loading saved results from {results_file}...")
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        print("Running experiments...")
        results = run_experiments(config)
        print(f"Saving results to {results_file}...")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

    plot_figure3(results, config)
    plot_figure3_linear(results, config)
    plot_figure3_paper_style(results, config)
    plot_figure3_combined(results, config)
    print_table1(results, config)

    print("\nAll done! Saved:")
    print("  figure3_reproduction.png       (log-log scale, full LISTA curves)")
    print("  figure3_reproduction_linear.png (linear scale, full LISTA curves)")
    print("  figure3_paper_style.png        (paper style — LISTA final points only)")
    print("  figure3_combined.png             (paper style, all conditions on one plot)")
    print("  table1_results.txt")
    print("  table1_results.csv")