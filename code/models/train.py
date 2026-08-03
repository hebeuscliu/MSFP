"""
Training pipeline for MSFP-MD with scaffold-based K-fold validation.

Supports:
  - Scaffold split (primary) and random stratified split (comparison)
  - Meta-learning with GradNorm-style dynamic loss weighting
  - Per-fold threshold tuning (F0.5 optimization)
  - Soft-voting ensemble across folds
  - Comprehensive metric reporting (AUC, AUPR, ACC, F1, MCC, Spec, Recall, Prec)
  - Optuna hyperparameter optimization
  - Ablation study framework
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, average_precision_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RESULTS_DIR, DATA_PROCESSED,
    MOLCLR_VEC_DIM, UNIMOL_VEC_DIM, OUTPUT_DIM, BATCH_SIZE,
    K_FOLDS, RANDOM_SEED, EARLY_STOP_PATIENCE, MAX_EPOCHS,
    DEFAULT_HPARAMS,
)
from models.fusion_model import MetaFusionMLP, build_ablation_variant
from scripts.scaffold_split import ScaffoldKFold

sns.set(style="whitegrid", font_scale=1.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Metric Helpers ──────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        return {k: 0.5 for k in [
            "auc", "acc", "precision", "recall", "f1", "specificity",
            "mcc", "aupr"
        ]}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "acc": accuracy_score(y_true, y_pred),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "aupr": average_precision_score(y_true, y_prob),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        beta: float = 0.5,
                        min_thresh: float = 0.05,
                        max_thresh: float = 0.95,
                        n_steps: int = 91) -> Tuple[float, float]:
    """Find threshold that maximizes F-beta score."""
    best_score, best_t = -1.0, 0.5
    for t in np.linspace(min_thresh, max_thresh, n_steps):
        preds = (y_prob >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if (beta**2 * prec + rec) == 0:
            f_beta = 0.0
        else:
            f_beta = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
        if f_beta > best_score:
            best_score, best_t = f_beta, float(t)
    return best_score, best_t


# ── Training Loop ───────────────────────────────────────────────────

def train_epoch(
    model: MetaFusionMLP,
    loader: DataLoader,
    criterion: nn.Module,
    opt_model: optim.Optimizer,
    opt_meta: optim.Optimizer,
    device: torch.device,
):
    """Single epoch with meta-learning (GradNorm-style weight update)."""
    model.train()
    shared_param = model.classifier[0].weight

    for xm, xu, y in loader:
        xm, xu, y = xm.to(device), xu.to(device), y.to(device)

        # ── Model update ──
        opt_model.zero_grad()
        opt_meta.zero_grad()

        output = model(xm, xu)
        loss_base = criterion(output, y).mean()
        (model.W_molclr * loss_base + model.W_unimol * loss_base).backward()
        opt_model.step()

        # ── Meta update (GradNorm) ──
        opt_meta.zero_grad()
        output_new = model(xm, xu)
        loss_new = criterion(output_new, y).mean()

        g_m = torch.autograd.grad(
            model.W_molclr * loss_new, shared_param,
            retain_graph=True, create_graph=True, allow_unused=True,
        )[0]
        g_u = torch.autograd.grad(
            model.W_unimol * loss_new, shared_param,
            retain_graph=True, create_graph=True, allow_unused=True,
        )[0]

        g_m_norm = g_m.norm() if g_m is not None else torch.tensor(0.0, device=device)
        g_u_norm = g_u.norm() if g_u is not None else torch.tensor(0.0, device=device)
        g_avg = (g_m_norm.detach() + g_u_norm.detach()) / 2.0

        loss_meta = (
            torch.abs(g_m_norm - g_avg) +
            torch.abs(g_u_norm - g_avg)
        )
        loss_meta.backward()
        opt_meta.step()

        # Clamp weights to stay positive
        with torch.no_grad():
            model.W_molclr.clamp_(min=0.01)
            model.W_unimol.clamp_(min=0.01)


def evaluate_auc(model, loader, device):
    """Quick AUC evaluation on a loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xm, xu, y in loader:
            out = model(xm.to(device), xu.to(device))
            preds.extend(out.cpu().numpy().flatten())
            labels.extend(y.numpy().flatten())
    labels = np.array(labels)
    preds = np.array(preds)
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


# ── Full K-Fold Run ─────────────────────────────────────────────────

def run_kfold_experiment(
    X_molclr: np.ndarray,
    X_unimol: np.ndarray,
    y: np.ndarray,
    smiles_list: List[str],
    hparams: Dict,
    split_method: str = "scaffold",
    model_variant: str = "full",
    output_dir: str = RESULTS_DIR,
    save_models: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Run a complete K-fold cross-validation experiment.

    Parameters
    ----------
    split_method : "scaffold" | "random"
        Scaffold-based splitting (primary) or stratified random (comparison).
    model_variant : str
        Ablation variant name (see build_ablation_variant).
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = hparams

    print(f"\n{'='*60}")
    print(f"K-Fold Experiment: split={split_method}, model={model_variant}")
    print(f"Data: {len(y)} samples (P={int(y.sum())}, N={int(len(y)-y.sum())})")
    print(f"{'='*60}")

    # ── Prepare tensors ──
    Xm_t = torch.from_numpy(X_molclr).float()
    Xu_t = torch.from_numpy(X_unimol).float()
    y_t = torch.from_numpy(y).float().view(-1, 1)

    # ── Create splits ──
    if split_method == "scaffold":
        kfold = ScaffoldKFold(n_splits=K_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
        splits = list(kfold.split(Xm_t, y_t.numpy(), smiles_list))
    else:
        kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)
        splits = list(kfold.split(Xm_t.numpy(), y))

    # ── Per-fold tracking ──
    fold_metrics = {
        "auc": [], "acc": [], "precision": [], "recall": [],
        "f1": [], "specificity": [], "mcc": [], "aupr": [],
    }
    fold_thresholds = []
    fold_weights_molclr = []
    fold_weights_unimol = []
    all_val_labels = []
    all_val_preds = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            n_pos_val = int(y_t[val_idx].sum())
            print(f"\n── Fold {fold+1}/{K_FOLDS} ── "
                  f"Train={len(train_idx)}, Val={len(val_idx)} "
                  f"(Val Pos={n_pos_val})")

        # Build DataLoaders
        train_loader = DataLoader(
            TensorDataset(Xm_t[train_idx], Xu_t[train_idx], y_t[train_idx]),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(Xm_t[val_idx], Xu_t[val_idx], y_t[val_idx]),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        # Build model
        model_kwargs = {
            "molclr_dim": MOLCLR_VEC_DIM, "unimol_dim": UNIMOL_VEC_DIM,
            "embed_dim": cfg["EMBED_DIM"], "output_dim": OUTPUT_DIM,
            "dropout_rate": cfg["DROPOUT"],
        }
        if model_variant in ("molclr_only", "unimol_only"):
            model_kwargs["input_dim"] = (
                MOLCLR_VEC_DIM if model_variant == "molclr_only"
                else UNIMOL_VEC_DIM
            )

        model = build_ablation_variant(model_variant, **model_kwargs).to(device)

        # Optimizers
        criterion = nn.BCELoss(reduction="none").to(device)

        # Only use meta-learning for the full model
        if model_variant == "full":
            opt_model = optim.Adam(
                [p for n, p in model.named_parameters() if "W_" not in n],
                lr=cfg["LEARNING_RATE"], weight_decay=1e-5,
            )
            opt_meta = optim.Adam(
                [model.W_molclr, model.W_unimol], lr=cfg["META_LR"]
            )
            use_meta = True
        else:
            opt_model = optim.Adam(
                model.parameters(), lr=cfg["LEARNING_RATE"], weight_decay=1e-5,
            )
            opt_meta = None
            use_meta = False

        # Training loop
        best_auc, patience = -1.0, 0
        best_state = None

        for epoch in range(cfg.get("EPOCHS", MAX_EPOCHS)):
            if use_meta:
                train_epoch(model, train_loader, criterion, opt_model,
                           opt_meta, device)
            else:
                # Standard training for ablation models
                model.train()
                for xm, xu, yb in train_loader:
                    xm, xu, yb = xm.to(device), xu.to(device), yb.to(device)
                    opt_model.zero_grad()
                    if model_variant in ("molclr_only", "unimol_only"):
                        loss = criterion(model(xm), yb).mean()
                    else:
                        loss = criterion(model(xm, xu), yb).mean()
                    loss.backward()
                    opt_model.step()

            cur_auc = evaluate_auc(model, val_loader, device)

            if cur_auc > best_auc + 1e-6:
                best_auc = cur_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    if verbose:
                        print(f"  Early stop @ epoch {epoch+1}, "
                              f"best val AUC={best_auc:.4f}")
                    break

        # Load best model
        model.load_state_dict(best_state)
        if save_models:
            torch.save(
                best_state,
                os.path.join(output_dir, f"model_{split_method}_fold{fold+1}.pth"),
            )

        # Record meta-weights for full model
        if model_variant == "full":
            fold_weights_molclr.append(model.W_molclr.item())
            fold_weights_unimol.append(model.W_unimol.item())

        # Evaluate on validation set
        model.eval()
        v_labels, v_probs = [], []
        with torch.no_grad():
            for xm, xu, yb in val_loader:
                xm, xu = xm.to(device), xu.to(device)
                if model_variant in ("molclr_only", "unimol_only"):
                    out = model(xm)
                else:
                    out = model(xm, xu)
                v_probs.extend(out.cpu().numpy().flatten())
                v_labels.extend(yb.numpy().flatten())

        v_labels = np.array(v_labels)
        v_probs = np.array(v_probs)

        # Threshold tuning
        _, best_t = find_best_threshold(v_labels, v_probs, beta=0.5)
        fold_thresholds.append(best_t)

        # Metrics
        m = compute_metrics(v_labels, v_probs, threshold=best_t)
        for k in fold_metrics:
            fold_metrics[k].append(m[k])

        all_val_labels.extend(v_labels)
        all_val_preds.extend(v_probs)

        if verbose:
            print(f"  Fold {fold+1} → AUC={m['auc']:.4f}, F1={m['f1']:.4f}, "
                  f"F0.5_thresh={best_t:.3f}")

    # ── Aggregate Results ──
    results = {
        "split_method": split_method,
        "model_variant": model_variant,
        "n_samples": len(y),
        "n_folds": K_FOLDS,
    }
    for k, vals in fold_metrics.items():
        vals_arr = np.array(vals)
        results[f"{k}_mean"] = float(np.mean(vals_arr))
        results[f"{k}_std"] = float(np.std(vals_arr))
        results[f"{k}_per_fold"] = [float(v) for v in vals_arr]

    results["threshold_mean"] = float(np.mean(fold_thresholds))
    results["threshold_per_fold"] = [float(t) for t in fold_thresholds]

    if fold_weights_molclr:
        results["W_molclr_mean"] = float(np.mean(fold_weights_molclr))
        results["W_unimol_mean"] = float(np.mean(fold_weights_unimol))

    # Pooled validation set metrics
    pooled_labels = np.array(all_val_labels)
    pooled_preds = np.array(all_val_preds)
    _, pooled_t = find_best_threshold(pooled_labels, pooled_preds, beta=0.5)
    pooled_m = compute_metrics(pooled_labels, pooled_preds, threshold=pooled_t)
    for k, v in pooled_m.items():
        results[f"pooled_{k}"] = float(v)

    # ── Save results ──
    results_path = os.path.join(output_dir,
                                f"results_{split_method}_{model_variant}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"Summary: {split_method} split, {model_variant}")
    print(f"{'='*60}")
    for k in ["auc", "aupr", "f1", "recall", "specificity", "mcc"]:
        print(f"  {k:12s}: {results[f'{k}_mean']:.4f} ± {results[f'{k}_std']:.4f}")
    print(f"{'='*60}")

    return results


# ── External Test Set Evaluation ───────────────────────────────────

def evaluate_external_test(
    model_dir: str,
    X_molclr_test: np.ndarray,
    X_unimol_test: np.ndarray,
    y_test: np.ndarray,
    model_variant: str = "full",
    hparams: Dict = DEFAULT_HPARAMS,
    output_dir: str = RESULTS_DIR,
) -> Dict:
    """
    Evaluate ensemble of K-fold models on a held-out external test set.
    Uses soft voting (average probability) across folds.
    """
    print(f"\n{'='*60}")
    print(f"External Test Evaluation ({model_variant})")
    print(f"{'='*60}")

    Xm_t = torch.from_numpy(X_molclr_test).float()
    Xu_t = torch.from_numpy(X_unimol_test).float()
    y_t = torch.from_numpy(y_test).float().view(-1, 1)

    test_loader = DataLoader(
        TensorDataset(Xm_t, Xu_t, y_t),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # Load models from each fold and ensemble
    fold_preds = []
    fold_models = []
    for fold in range(1, K_FOLDS + 1):
        model_path = os.path.join(model_dir, f"model_scaffold_fold{fold}.pth")
        if not os.path.exists(model_path):
            print(f"  ⚠ Model for fold {fold} not found at {model_path}, skipping")
            continue

        model_kwargs = {
            "molclr_dim": MOLCLR_VEC_DIM, "unimol_dim": UNIMOL_VEC_DIM,
            "embed_dim": hparams["EMBED_DIM"], "output_dim": OUTPUT_DIM,
            "dropout_rate": hparams["DROPOUT"],
        }
        if model_variant in ("molclr_only", "unimol_only"):
            model_kwargs["input_dim"] = (
                MOLCLR_VEC_DIM if model_variant == "molclr_only"
                else UNIMOL_VEC_DIM
            )
        model = build_ablation_variant(model_variant, **model_kwargs).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        fold_models.append(model)

        probs = []
        with torch.no_grad():
            for xm, xu, _ in test_loader:
                xm, xu = xm.to(device), xu.to(device)
                if model_variant in ("molclr_only", "unimol_only"):
                    out = model(xm)
                else:
                    out = model(xm, xu)
                probs.extend(out.cpu().numpy().flatten())
        fold_preds.append(np.array(probs))

    if not fold_preds:
        print("  No models found!")
        return {}

    # Soft voting ensemble
    ensemble_probs = np.mean(fold_preds, axis=0)
    y_true = y_test.flatten()

    metrics = compute_metrics(y_true, ensemble_probs)
    _, best_t = find_best_threshold(y_true, ensemble_probs, beta=0.5)
    metrics_at_t = compute_metrics(y_true, ensemble_probs, threshold=best_t)

    print(f"\n  Ensemble (n_models={len(fold_preds)}, threshold={best_t:.3f}):")
    for k in ["auc", "aupr", "f1", "recall", "specificity", "mcc"]:
        print(f"    {k}: {metrics_at_t[k]:.4f}")

    # Save
    results = {
        "n_models": len(fold_preds),
        "threshold": float(best_t),
        "metrics_uncalibrated": {k: float(v) for k, v in metrics.items()},
        "metrics_calibrated": {k: float(v) for k, v in metrics_at_t.items()},
    }
    with open(os.path.join(output_dir, "external_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(output_dir, "ensemble_test_preds.npy"), ensemble_probs)

    # ROC plot
    try:
        fpr, tpr, _ = roc_curve(y_true, ensemble_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC={metrics['auc']:.4f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("External Test ROC (Ensemble)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "external_test_roc.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    return results


# ── Optuna Hyperparameter Search ────────────────────────────────────

def optuna_search(
    X_molclr: np.ndarray,
    X_unimol: np.ndarray,
    y: np.ndarray,
    smiles_list: List[str],
    split_method: str = "scaffold",
    n_trials: int = 40,
    output_dir: str = RESULTS_DIR,
) -> Dict:
    """Run Optuna hyperparameter search using a single scaffold fold."""

    Xm_t = torch.from_numpy(X_molclr).float()
    Xu_t = torch.from_numpy(X_unimol).float()
    y_t = torch.from_numpy(y).float().view(-1, 1)

    # Get first split
    if split_method == "scaffold":
        kfold = ScaffoldKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = kfold.split(Xm_t, y_t.numpy(), smiles_list)
    else:
        kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)
        splits = kfold.split(Xm_t.numpy(), y)
    train_idx, val_idx = next(splits)

    train_loader = DataLoader(
        TensorDataset(Xm_t[train_idx], Xu_t[train_idx], y_t[train_idx]),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(Xm_t[val_idx], Xu_t[val_idx], y_t[val_idx]),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    def objective(trial):
        embed_dim = trial.suggest_categorical("EMBED_DIM", [128, 192, 256, 320, 384])
        dropout = trial.suggest_float("DROPOUT", 0.05, 0.5)
        lr = trial.suggest_float("LEARNING_RATE", 1e-6, 5e-4, log=True)
        meta_lr = trial.suggest_float("META_LR", 1e-5, 1e-2, log=True)

        model = MetaFusionMLP(
            MOLCLR_VEC_DIM, UNIMOL_VEC_DIM, embed_dim, OUTPUT_DIM, dropout,
        ).to(device)
        criterion = nn.BCELoss(reduction="none").to(device)
        opt_model = optim.Adam(
            [p for n, p in model.named_parameters() if "W_" not in n],
            lr=lr, weight_decay=1e-5,
        )
        opt_meta = optim.Adam([model.W_molclr, model.W_unimol], lr=meta_lr)

        best_auc = -1.0
        patience = 0
        for epoch in range(50):
            train_epoch(model, train_loader, criterion, opt_model, opt_meta, device)
            cur_auc = evaluate_auc(model, val_loader, device)

            trial.report(cur_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if cur_auc > best_auc + 1e-6:
                best_auc = cur_auc
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break
        return float(best_auc)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_warmup_steps=5),
    )
    print(f"\nOptuna search ({n_trials} trials, {split_method} split)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["EPOCHS"] = MAX_EPOCHS
    best_params["META_LR"] = float(best_params.get("META_LR", 0.001))
    best_params["LEARNING_RATE"] = float(best_params.get("LEARNING_RATE", 1e-4))
    best_params["EMBED_DIM"] = int(best_params.get("EMBED_DIM", 256))
    best_params["DROPOUT"] = float(best_params.get("DROPOUT", 0.13))

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    return best_params


# ── Ablation Study Runner ───────────────────────────────────────────

def run_ablation_study(
    X_molclr: np.ndarray,
    X_unimol: np.ndarray,
    y: np.ndarray,
    smiles_list: List[str],
    hparams: Dict,
    split_method: str = "scaffold",
    output_dir: str = RESULTS_DIR,
) -> pd.DataFrame:
    """Run all ablation variants and return comparison DataFrame."""

    variants = [
        "full", "no_lstm", "no_attention", "no_gate",
        "simple_concat", "molclr_only", "unimol_only",
    ]

    all_results = []
    for variant in variants:
        print(f"\n{'#'*60}")
        print(f"# Ablation: {variant}")
        print(f"{'#'*60}")

        var_dir = os.path.join(output_dir, f"ablation_{variant}")
        os.makedirs(var_dir, exist_ok=True)

        results = run_kfold_experiment(
            X_molclr, X_unimol, y, smiles_list,
            hparams=hparams,
            split_method=split_method,
            model_variant=variant,
            output_dir=var_dir,
            save_models=True,
            verbose=True,
        )
        all_results.append(results)

    # Build comparison table
    rows = []
    key_metrics = ["auc", "aupr", "f1", "recall", "specificity", "mcc"]
    for r in all_results:
        row = {"variant": r["model_variant"]}
        for m in key_metrics:
            row[m] = f"{r[f'{m}_mean']:.4f}±{r[f'{m}_std']:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "ablation_comparison.csv"), index=False)
    print(f"\n{'='*60}")
    print("Ablation Study Summary")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    return df


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MSFP-MD Training Pipeline"
    )
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "ablation", "optuna", "evaluate"],
                        help="Run mode")
    parser.add_argument("--split", type=str, default="scaffold",
                        choices=["scaffold", "random"],
                        help="Data split method")
    parser.add_argument("--variant", type=str, default="full",
                        help="Model variant (for train/evaluate)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing .npy feature files")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV file with SMILES for scaffold computation")
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                        help="Output directory for results")
    parser.add_argument("--optuna-trials", type=int, default=40)

    args = parser.parse_args()

    # Determine data directory
    data_dir = args.data_dir or os.path.join(DATA_PROCESSED, "property_matched")

    print(f"MSFP-MD Training Pipeline")
    print(f"  Mode: {args.mode}")
    print(f"  Split: {args.split}")
    print(f"  Device: {device}")

    # Load data
    X_molclr = np.load(os.path.join(data_dir, "X_molclr_train.npy"))
    X_unimol = np.load(os.path.join(data_dir, "unimol_fixed_train.npy"))
    y = np.load(os.path.join(data_dir, "Y_full_train.npy")).flatten().astype(np.float32)

    # Load SMILES for scaffold computation
    csv_path = args.csv or os.path.join(data_dir, "train.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        smiles_list = df["smiles"].tolist()
    else:
        print(f"Warning: CSV {csv_path} not found. Using empty SMILES.")
        smiles_list = [""] * len(y)

    if args.mode == "train":
        run_kfold_experiment(
            X_molclr, X_unimol, y, smiles_list,
            hparams=DEFAULT_HPARAMS,
            split_method=args.split,
            model_variant=args.variant,
            output_dir=args.output_dir,
        )

    elif args.mode == "ablation":
        run_ablation_study(
            X_molclr, X_unimol, y, smiles_list,
            hparams=DEFAULT_HPARAMS,
            split_method=args.split,
            output_dir=args.output_dir,
        )

    elif args.mode == "optuna":
        best = optuna_search(
            X_molclr, X_unimol, y, smiles_list,
            split_method=args.split,
            n_trials=args.optuna_trials,
            output_dir=args.output_dir,
        )
        print(f"\nBest params: {json.dumps(best, indent=2)}")

    elif args.mode == "evaluate":
        Xm_test = np.load(os.path.join(data_dir, "X_molclr_test.npy"))
        Xu_test = np.load(os.path.join(data_dir, "unimol_fixed_test.npy"))
        y_test = np.load(os.path.join(data_dir, "Y_full_test.npy")).flatten()
        evaluate_external_test(
            args.output_dir, Xm_test, Xu_test, y_test,
            model_variant=args.variant,
            output_dir=args.output_dir,
        )
