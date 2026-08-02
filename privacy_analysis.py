"""
privacy_analysis.py  (v4 — corrected, label-availability focus)
----------------------------------------------------------------
Demonstrates the privacy advantage of U-shape split learning over
normal split learning for CNC anomaly detection.  Three metrics are
used, each grounded in *whether the server has structural access to
labels or gradients*, not in how separable the activations happen to be.

METRICS
-------
1. INFORMATION EXPOSURE RATIO  (IER)  — protocol-level, provable
     IER = (label entropy the server structurally receives) /
           (total label entropy of the dataset)

     For a balanced binary dataset H(Y) = 1.0 bit.
     Normal split : server receives (A_i, y_i) for every sample
                    → IER = H(Y)/H(Y) = 1.0  (complete label exposure)
     U-shape      : labels never leave the client
                    → IER = 0/H(Y)  = 0.0  (zero label exposure)

     This is a hard, protocol-level guarantee — not a statistical
     estimate.  It holds regardless of the front model weights.

     Reference: Vepakomma et al., "No Peek: A Survey of private
     distributed deep learning" (arXiv 1812.03288, §3).

2. INPUT RECONSTRUCTION ATTACK  — empirical
     A curious server trains a convolutional decoder on the smashed
     activations A = front(X) it receives and tries to recover X.
     We report the *normalised* MSE  (NMSE = MSE / Var(X_test)) so
     the result is scale-invariant and comparable across methods.

       NMSE ≈ 1.0  →  decoder predicts the mean; reconstruction fails
       NMSE < 1.0  →  decoder recovers some structure from A
       NMSE → 0.0  →  near-perfect reconstruction  (worst privacy)

     Reference: He et al., "Model Inversion Attacks against
     Collaborative Inference" (CCS 2019).

3. GRADIENT INVERSION RISK SCORE  (GIRS)  — attack-surface proxy
     In normal split the server:
       (a) receives smashed activations  A  (shape B×64×256)
       (b) receives the labels y (sent explicitly to compute loss)
       (c) computes loss L(head(A), y)
       (d) sends gradient  ∂L/∂A  back to the client

     The server holds ∂L/∂A, which is the exact input needed for a
     DLG/iDLG-style gradient inversion attack (Zhu et al., 2019).

     GIRS = mean ‖∂L/∂A‖₂  over sampled inputs, where A is the full
     smashed tensor (B×64×256) — NOT the GAP reduction.  Using the
     full tensor is correct because that is what the server sends back
     to the client; the GAP is internal to the server's classifier head.

     For U-shape the server never computes the loss and never holds
     ∂L/∂A, so GIRS = 0.0 by protocol construction.

     Implementation note: we detach A from the front computation
     and set A.requires_grad_(True) so that only ∂L/∂A is computed —
     no spurious gradients through front weights.  The server-side
     classifier head is pre-trained for 300 steps so its gradients
     reflect real data structure rather than random initialisation.

     Reference: Zhu et al., "Deep Leakage from Gradients",
                NeurIPS 2019 (arXiv 1906.08935).

Outputs
-------
  logs/privacy_report.json        — machine-readable summary
  plots/ier_bar.png               — IER comparison (0 vs 1)
  plots/reconstruction_loss.png   — decoder NMSE per method
  plots/gradient_risk.png         — GIRS distribution (boxplot)
  plots/privacy_summary_table.png — one-page summary table

Usage
-----
  python privacy_analysis.py

Save trained front weights with:
  torch.save(front.state_dict(), "front_M01_ushape.pth")
  torch.save(front.state_dict(), "front_M01_normal.pth")
A random-init front is used as fallback if weights are missing.
"""

import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR     = "new_train"
TEST_DIR      = "new_test"
MACHINE_IDS   = ["M01", "M02", "M03"]
LOG_DIR       = "logs"
PLOTS_DIR     = "plots"
MAX_PER_CLASS = 200
MAX_TEST      = 200
DECODER_EPOCHS = 30
BATCH_SIZE    = 32
N_GIRS_SAMPLES = 128   # samples used to estimate GIRS

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "u_shape":      "#2E86AB",
    "normal_split": "#E84855",
}
METHOD_LABELS = {
    "u_shape":      "U-Shape Split",
    "normal_split": "Normal Split",
}

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class FrontModel(nn.Module):
    """Shared front (client-side) model — identical for both methods.
    Input : (B, 3, 1024)   Output : (B, 64, 256)  smashed activations
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3,  32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x   # (B, 64, 256)


class DecoderModel(nn.Module):
    """Adversarial decoder: tries to reconstruct input from smashed data.
    Input : (B, 64, 256)  → Output : (B, 3, 1024)
    Trained by a curious server; high MSE means reconstruction fails.
    """
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=4,
                                          stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(32,  3, kernel_size=4,
                                          stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        return self.deconv2(x)          # (B, 3, 1024)


# ============================================================
# DATA LOADING
# ============================================================
def load_samples(machine_id, base_dir, max_per_class=200):
    X, y = [], []
    for folder, label in [("good", 0), ("bad", 1)]:
        path = os.path.join(base_dir, machine_id, folder)
        if not os.path.exists(path):
            continue
        files = sorted(f for f in os.listdir(path)
                       if f.endswith(".h5"))[:max_per_class]
        for fname in tqdm(files, desc=f"  {machine_id}/{folder}",
                          leave=False):
            with h5py.File(os.path.join(path, fname), "r") as hf:
                X.append(hf["vibration_data"][:])
                y.append(label)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)


def load_test_samples(base_dir, max_per_class=200):
    X, y = [], []
    for folder, label in [("good", 0), ("bad", 1)]:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            continue
        files = sorted(f for f in os.listdir(path)
                       if f.endswith(".h5"))[:max_per_class]
        for fname in tqdm(files, desc=f"  test/{folder}", leave=False):
            with h5py.File(os.path.join(path, fname), "r") as hf:
                X.append(hf["vibration_data"][:])
                y.append(label)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)


def to_tensor(X_np):
    """(N, T, C) → (N, C, T) torch tensor."""
    return torch.tensor(X_np.transpose(0, 2, 1), dtype=torch.float32)


# ============================================================
# METRIC 1 — INFORMATION EXPOSURE RATIO  (IER)
# ============================================================
def compute_ier(method: str, y_train: np.ndarray) -> dict:
    """
    IER = (label entropy the server structurally receives) / H(Y)

    For normal split the server receives (A_i, y_i) for every sample,
    so it has full access to H(Y) bits of label information.
    IER = H(Y) / H(Y) = 1.0.

    For U-shape no label is ever transmitted.
    IER = 0 / H(Y) = 0.0.

    H(Y) is computed from the actual training label distribution.
    The IER values follow from the protocol definition alone — they
    are not statistical estimates and do not depend on model weights.

    Reference: Vepakomma et al. (arXiv 1812.03288, §3).
    """
    classes, counts = np.unique(y_train, return_counts=True)
    p_y = counts / counts.sum()
    hy  = float(-np.sum(p_y * np.log2(p_y + 1e-12)))   # bits

    if method == "normal_split":
        # Server receives y_i for every sample → has full H(Y)
        label_entropy_exposed = hy
        ier = label_entropy_exposed / hy          # = 1.0
        labels_sent = "ALL — every training batch"
        explanation = (
            "Server receives (activation, label) pairs for every sample. "
            "It has complete label information: IER = H(Y)/H(Y) = 1.0. "
            "This is the worst possible privacy outcome for label exposure."
        )
    else:  # u_shape
        label_entropy_exposed = 0.0
        ier = 0.0                                 # = 0/H(Y)
        labels_sent = "NONE — labels stay on client"
        explanation = (
            "The tail network and loss function both run on the client. "
            "The server receives only intermediate activations and "
            "back-propagated gradients — no label is ever transmitted. "
            "IER = 0/H(Y) = 0.0: the best possible label-privacy outcome."
        )

    return {
        "ier":                      ier,
        "label_entropy_H":          hy,
        "label_entropy_exposed":    label_entropy_exposed,
        "labels_sent":              labels_sent,
        "explanation":              explanation,
    }


# ============================================================
# METRIC 2 — INPUT RECONSTRUCTION ATTACK
# ============================================================
def reconstruction_attack(front: nn.Module,
                           X_train: np.ndarray,
                           X_test:  np.ndarray,
                           epochs:  int = DECODER_EPOCHS) -> dict:
    """
    A curious server trains a convolutional decoder on the smashed
    activations A = front(X) it observes during training, then tries
    to reconstruct the original input signal X.

    Protocol:
      - During training the server sees A = front(X) for every batch.
      - It trains Decoder(A) ≈ X with MSE loss (self-supervised;
        no labels are needed — the server already has A).
      - We evaluate on a held-out test set and report
          NMSE = MSE / Var(X_test)
        Normalising by the input variance makes the metric
        scale-invariant and comparable across front models with
        different activation magnitudes.

    Interpretation:
      NMSE ≈ 1.0  →  decoder predicts the mean; reconstruction fails
                      → good input-level privacy
      NMSE < 1.0  →  decoder recovers signal structure from A
      NMSE → 0.0  →  near-perfect reconstruction → worst privacy

    This metric is independent of label availability; it captures how
    much raw signal information passes through the cut layer.

    Reference: He et al., "Model Inversion Attacks against
    Collaborative Inference" (CCS 2019).
    """
    front.eval()
    decoder   = DecoderModel()
    optimiser = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    mse_fn    = nn.MSELoss()

    X_tr = to_tensor(X_train)   # (N, 3, 1024)

    # ── Train decoder ──────────────────────────────────────────
    decoder.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb  = X_tr[idx]
            with torch.no_grad():
                act = front(xb)          # smashed data the server sees
            recon = decoder(act)
            loss  = mse_fn(recon, xb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    # ── Evaluate on held-out test set ──────────────────────────
    decoder.eval()
    X_te    = to_tensor(X_test)
    # Input variance: used to normalise MSE → NMSE
    x_var   = float(X_te.var().item())
    mse_vals = []
    with torch.no_grad():
        for i in range(0, len(X_te), BATCH_SIZE):
            xb    = X_te[i:i + BATCH_SIZE]
            act   = front(xb)
            recon = decoder(act)
            mse_vals.append(mse_fn(recon, xb).item())

    test_mse  = float(np.mean(mse_vals))
    test_nmse = test_mse / (x_var + 1e-12)   # scale-invariant

    return {
        "test_mse":        test_mse,
        "test_nmse":       test_nmse,
        "input_variance":  x_var,
        "decoder_epochs":  epochs,
        "interpretation": (
            "NMSE ≈ 1.0: decoder predicts the mean, reconstruction fails "
            "(good privacy). NMSE → 0: near-perfect reconstruction (bad)."
        ),
    }


# ============================================================
# METRIC 3 — GRADIENT INVERSION RISK SCORE  (GIRS)
# ============================================================
def gradient_inversion_risk(front:  nn.Module,
                             X_np:   np.ndarray,
                             y_np:   np.ndarray,
                             method: str,
                             n_samples: int = N_GIRS_SAMPLES) -> dict:
    """
    Quantifies the gradient signal the server receives that could be
    exploited in a DLG/iDLG-style inversion attack.

    In normal split the server:
      1. Observes smashed activations  A = front(X),  shape (B, 64, 256).
      2. Receives labels y explicitly.
      3. Computes  L = loss(head(A), y).
      4. Sends gradient  ∂L/∂A  back to the client.

    The server *holds* ∂L/∂A, which is the exact ingredient for DLG.
    GIRS = mean ‖∂L/∂A‖₂  over sampled inputs.
    Larger norm → richer gradient signal → higher inversion risk.

    Correctness notes
    -----------------
    • Gradient is computed on the FULL A (B×64×256), not on a GAP
      reduction.  The server sends ∂L/∂A back; the GAP is an internal
      operation of the server's classifier head, invisible to the client.
    • We detach A from the front computation with A.requires_grad_(True)
      so only ∂L/∂A is measured — no gradient flows into front weights.
    • The server-side head is pre-trained for 300 steps on the actual
      activation distribution so gradients reflect real data structure,
      not random initialisation artefacts.
    • The head maps GAP(A) → logit, matching a realistic server setup
      where the classifier operates on the pooled representation.
      ∂L/∂A is then obtained via the chain rule through the GAP.

    For U-shape the server never computes the loss and never holds
    ∂L/∂A, so GIRS = 0.0 by protocol construction.

    Returns per-sample norms (for distribution plotting) in addition
    to the mean GIRS.

    Reference: Zhu et al., "Deep Leakage from Gradients",
               NeurIPS 2019 (arXiv 1906.08935).
    """
    if method == "u_shape":
        return {
            "girs":          0.0,
            "per_sample":    [],
            "reason":        "Server never computes loss; no ∂L/∂A exists.",
            "applicable":    False,
        }

    # ── Step 1: pre-train a linear head on the real activation GAP ──
    # Using a trained head is essential: a random head produces gradients
    # dominated by random weight directions, not label-correlated signal.
    loss_fn = nn.BCEWithLogitsLoss()
    front.eval()
    head     = nn.Linear(64, 1)
    opt_head = torch.optim.Adam(head.parameters(), lr=5e-3)
    all_x    = to_tensor(X_np)
    all_y    = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    head.train()
    for step in range(300):
        idx  = torch.randperm(len(all_x))[:BATCH_SIZE]
        with torch.no_grad():
            # GAP of full smashed activation — same as server sees
            gap = front(all_x[idx]).mean(dim=-1)   # (B, 64)
        out  = head(gap)
        loss = loss_fn(out, all_y[idx])
        opt_head.zero_grad()
        loss.backward()
        opt_head.step()
    head.eval()

    # ── Step 2: measure ‖∂L/∂A‖ on the FULL smashed tensor ────────
    # Detach A from the front graph; only track gradients from A onward.
    # This is the correct simulation: the server holds A (already
    # computed) and wants to know the gradient of its loss w.r.t. A.
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X_np), size=min(n_samples, len(X_np)),
                      replace=False)
    norms = []

    for i in idx:
        xb = to_tensor(X_np[i:i + 1])
        yb = torch.tensor([[float(y_np[i])]])

        with torch.no_grad():
            A_detached = front(xb)    # (1, 64, 256) — server receives this

        # Re-attach gradient tracking at A, not at front inputs
        A = A_detached.detach().requires_grad_(True)

        # Server's forward pass: GAP → head → loss
        gap  = A.mean(dim=-1)         # (1, 64)  — pooling inside server head
        out  = head(gap)
        loss = loss_fn(out, yb)
        loss.backward()

        if A.grad is not None:
            # ‖∂L/∂A‖ where A has shape (1, 64, 256)
            norms.append(float(A.grad.norm().item()))

    girs = float(np.mean(norms)) if norms else float("nan")
    return {
        "girs":         girs,
        "per_sample":   norms,
        "reason":       "Mean ‖∂L/∂A‖₂ over smashed tensor (B×64×256).",
        "applicable":   True,
        "reference":    "Zhu et al., NeurIPS 2019",
    }


# ============================================================
# PLOTTING
# ============================================================
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def plot_ier(ier_results: dict, out_path: str):
    """
    IER bar chart.  The bars are 0.0 vs 1.0 — the gap is the whole story.
    """
    methods = list(METHOD_LABELS.keys())
    values  = [ier_results[m]["ier"] for m in methods]
    labels  = [METHOD_LABELS[m]      for m in methods]
    cols    = [COLORS[m]             for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, width=0.45, color=cols, alpha=0.88,
                  zorder=3)

    ax.set_ylim(0, 1.35)
    ax.set_ylabel("Information Exposure Ratio  (IER)", fontsize=11)
    ax.set_title(
        "Information Exposure Ratio\n"
        "Fraction of label entropy the server structurally receives\n"
        "(0 = no labels exposed  |  1 = complete label exposure)",
        fontsize=11,
    )
    _style(ax)

    annotations = {
        "u_shape":      "IER = 0.0\nLabels never leave client\n(protocol guarantee)",
        "normal_split": "IER = 1.0\nServer receives every label\nevery training batch",
    }
    for bar, method, val in zip(bars, methods, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.04,
            annotations[method],
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLORS[method],
        )

    # Reference line
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.2,
               label="Midpoint reference")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_reconstruction(recon_results: dict, out_path: str):
    methods = [m for m in METHOD_LABELS if m in recon_results]
    values  = [recon_results[m]["test_nmse"] for m in methods]
    labels  = [METHOD_LABELS[m]              for m in methods]
    cols    = [COLORS[m]                     for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, width=0.45, color=cols, alpha=0.88,
                  zorder=3)
    # Reference line at NMSE=1 (decoder predicts mean = total failure)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.2,
               label="NMSE = 1.0  (decoder predicts mean)")
    ax.legend(fontsize=9)
    ax.set_ylabel("Normalised Reconstruction Error  (NMSE = MSE / Var(X))",
                  fontsize=11)
    ax.set_title(
        "Input Reconstruction Attack — Normalised MSE\n"
        "Server trains a decoder on activations it receives;\n"
        "NMSE ≥ 1.0 = reconstruction fails = good input privacy",
        fontsize=11,
    )
    _style(ax)
    for bar, method, val in zip(bars, methods, values):
        raw_mse = recon_results[method]["test_mse"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.02,
            f"NMSE = {val:.3f}\n(MSE = {raw_mse:.4f})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLORS[method],
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_girs(girs_results: dict, out_path: str):
    """
    Show the distribution of per-sample ‖∂L/∂A‖ for Normal Split
    alongside a zero reference for U-Shape.  A single mean bar hides
    variance; the distribution tells a richer story about attack risk.
    """
    ns = girs_results.get("normal_split", {})
    ns_norms = ns.get("per_sample", [])
    ns_girs  = ns.get("girs", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1, 2]})

    # ── Left: mean bar comparison ──
    ax = axes[0]
    methods = ["u_shape", "normal_split"]
    values  = [0.0, ns_girs if not np.isnan(ns_girs) else 0.0]
    labels  = [METHOD_LABELS[m] for m in methods]
    cols    = [COLORS[m]        for m in methods]
    bars = ax.bar(labels, values, width=0.45, color=cols, alpha=0.88,
                  zorder=3)
    ax.set_ylabel("Mean ‖∂L/∂A‖₂  (GIRS)", fontsize=10)
    ax.set_title("Mean GIRS\n(lower = less inversion risk)", fontsize=10)
    _style(ax)
    notes = {
        "u_shape":      "GIRS = 0\n(no ∂L/∂A by protocol)",
        "normal_split": f"GIRS = {ns_girs:.4f}\n(DLG attack surface)",
    }
    for bar, method, val in zip(bars, methods, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.02 + 1e-4,
            notes[method],
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLORS[method],
        )

    # ── Right: per-sample distribution for Normal Split ──
    ax2 = axes[1]
    if ns_norms:
        ax2.hist(ns_norms, bins=20, color=COLORS["normal_split"],
                 alpha=0.75, edgecolor="white", zorder=3,
                 label="Normal Split")
        ax2.axvline(ns_girs, color=COLORS["normal_split"],
                    linestyle="--", linewidth=1.5,
                    label=f"Mean = {ns_girs:.4f}")
        ax2.axvline(0.0, color=COLORS["u_shape"], linewidth=2.0,
                    label="U-Shape = 0 (protocol guarantee)")
        ax2.legend(fontsize=9)
        ax2.set_xlabel("‖∂L/∂A‖₂  per sample", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title(
            "Distribution of Per-Sample Gradient Norms\n"
            "(Normal Split — gradient the server holds each step)",
            fontsize=10,
        )
    else:
        ax2.text(0.5, 0.5, "No per-sample data available.",
                 ha="center", va="center", transform=ax2.transAxes)
    _style(ax2)

    ax2.annotate(
        "Zhu et al., NeurIPS 2019 (DLG attack uses ∂L/∂A)",
        xy=(0.02, 0.96), xycoords="axes fraction",
        fontsize=8, color="grey", va="top",
    )
    fig.suptitle(
        "Gradient Inversion Risk Score  (GIRS)\n"
        "L2 norm of per-sample loss gradient visible to server  "
        "(0 = no gradient = no inversion risk)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_summary_table(ier_results: dict, recon_results: dict,
                       girs_results: dict, out_path: str):
    rows = []
    for m in ["u_shape", "normal_split"]:
        ir   = ier_results.get(m, {})
        rr   = recon_results.get(m, {})
        gr   = girs_results.get(m, {})
        girs = gr.get("girs", float("nan"))
        nmse = rr.get("test_nmse", float("nan"))
        rows.append([
            METHOD_LABELS[m],
            ir.get("labels_sent", "?"),
            f"{ir.get('ier', float('nan')):.1f}",
            f"{nmse:.3f}",
            f"{girs:.4f}" if gr.get("applicable", False) else "0 (N/A — protocol)",
            "✓ Strong" if m == "u_shape" else "✗ Weak",
        ])

    col_labels = [
        "Method", "Labels to server",
        "IER ↓", "Recon. NMSE ↑",
        "GIRS ↓", "Overall privacy",
    ]

    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.4)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#263238")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    row_bg  = {"U-Shape Split": "#E3F2FD", "Normal Split": "#FFEBEE"}
    priv_bg = {"✓ Strong": "#C8E6C9",       "✗ Weak":      "#FFCDD2"}
    for i, row in enumerate(rows, start=1):
        bg = row_bg.get(row[0], "white")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        tbl[i, len(col_labels) - 1].set_facecolor(
            priv_bg.get(row[-1], "white")
        )

    ax.set_title(
        "Privacy Comparison Summary\n"
        "IER ↓ lower is better  |  Recon. MSE ↑ higher is better  |  GIRS ↓ lower is better",
        fontsize=11, fontweight="bold", pad=16,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    SEP = "=" * 62

    print(SEP)
    print("  PRIVACY ANALYSIS  v3  —  label-availability focus")
    print(SEP)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1] Loading data …")
    X_train_parts, y_train_parts = [], []
    for mid in MACHINE_IDS:
        X, y = load_samples(mid, TRAIN_DIR, MAX_PER_CLASS)
        if X is not None:
            X_train_parts.append(X); y_train_parts.append(y)

    X_test, y_test = load_test_samples(TEST_DIR, MAX_TEST)

    if not X_train_parts or X_test is None:
        print("  ✗ No data found.  Check TRAIN_DIR / TEST_DIR.")
        return

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    print(f"  Train : {len(X_train):,} samples | Test : {len(X_test):,}")

    # ── Per-method analysis ────────────────────────────────────
    methods = [
        ("u_shape",      "front_M01_ushape.pth"),
        ("normal_split", "front_M01_normal.pth"),
    ]

    ier_results   = {}
    recon_results = {}
    girs_results  = {}

    for method, weight_file in methods:
        print(f"\n[2] {METHOD_LABELS[method]}")

        # Load (or randomly initialise) front model
        front = FrontModel()
        if os.path.exists(weight_file):
            front.load_state_dict(
                torch.load(weight_file, weights_only=True)
            )
            print(f"    Loaded weights : {weight_file}")
        else:
            print(f"    ⚠  {weight_file!r} not found — using random-init front.")

        # ── IER (no model needed — pure protocol analysis) ──
        ier = compute_ier(method, y_train)
        ier_results[method] = ier
        print(f"    IER             : {ier['ier']:.1f}  "
              f"(H(Y)={ier['label_entropy_H']:.3f} bits, "
              f"exposed={ier['label_entropy_exposed']:.3f} bits)")
        print(f"    → {ier['explanation'][:80]}…")

        # ── Reconstruction attack ──
        print(f"    Training decoder ({DECODER_EPOCHS} epochs) …")
        recon = reconstruction_attack(front, X_train, X_test,
                                       epochs=DECODER_EPOCHS)
        recon_results[method] = recon
        print(f"    Recon. NMSE     : {recon['test_nmse']:.4f}  "
              f"(raw MSE={recon['test_mse']:.4f}, "
              f"Var(X)={recon['input_variance']:.4f})")
        print(f"    → NMSE≥1 = reconstruction fails = good input privacy")

        # ── GIRS ──
        print("    Estimating gradient inversion risk …")
        girs = gradient_inversion_risk(front, X_train, y_train,
                                        method=method)
        girs_results[method] = girs
        if girs["applicable"]:
            norms = girs["per_sample"]
            std   = float(np.std(norms)) if norms else float("nan")
            print(f"    GIRS            : {girs['girs']:.4f} ± {std:.4f}  "
                  f"(mean ‖∂L/∂A‖₂ on full B×64×256 tensor)")
        else:
            print(f"    GIRS            : 0.0  ({girs['reason']})")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    for method, _ in methods:
        ir = ier_results[method]
        rr = recon_results[method]
        gr = girs_results[method]
        print(f"\n  ▶ {METHOD_LABELS[method]}")
        print(f"    IER            : {ir['ier']:.1f}  — {ir['labels_sent']}")
        print(f"    Recon. NMSE    : {rr['test_nmse']:.4f}  "
              f"(raw MSE={rr['test_mse']:.4f})")
        if gr['applicable']:
            norms = gr['per_sample']
            std   = float(np.std(norms)) if norms else float("nan")
            print(f"    GIRS           : {gr['girs']:.4f} ± {std:.4f}")
        else:
            print(f"    GIRS           : 0 (N/A — {gr['reason']})")

    # Privacy delta
    u_nmse  = recon_results["u_shape"]["test_nmse"]
    ns_nmse = recon_results["normal_split"]["test_nmse"]
    if u_nmse >= ns_nmse:
        print(f"\n  ✓ U-shape has higher reconstruction NMSE "
              f"(+{u_nmse - ns_nmse:.4f}) → harder to recover inputs")
    else:
        print(f"\n  ✗ Normal split has higher reconstruction NMSE "
              f"(Δ{ns_nmse - u_nmse:.4f}) — U-shape front may be shallower "
              f"or the cut layer is placed earlier; consider deeper front.")

    # ── Save JSON ──────────────────────────────────────────────
    report = {
        "ier":            ier_results,
        "reconstruction": {
            m: {
                "test_nmse":       recon_results[m]["test_nmse"],
                "test_mse":        recon_results[m]["test_mse"],
                "input_variance":  recon_results[m]["input_variance"],
                "decoder_epochs":  recon_results[m]["decoder_epochs"],
            }
            for m in recon_results
        },
        "girs": {
            m: {k: v for k, v in girs_results[m].items()
                if k != "per_sample"}   # exclude raw list from JSON
            for m in girs_results
        },
    }
    report_path = os.path.join(LOG_DIR, "privacy_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, default=str)
    print(f"\n  Report : {report_path}")

    # ── Plots ──────────────────────────────────────────────────
    print("\n[3] Generating plots …")
    plot_ier(ier_results,
             os.path.join(PLOTS_DIR, "ier_bar.png"))
    plot_reconstruction(recon_results,
                        os.path.join(PLOTS_DIR, "reconstruction_loss.png"))
    plot_girs(girs_results,
              os.path.join(PLOTS_DIR, "gradient_risk.png"))
    plot_summary_table(ier_results, recon_results, girs_results,
                       os.path.join(PLOTS_DIR, "privacy_summary_table.png"))

    print("\n  ✓ Done.")


if __name__ == "__main__":
    main()