#!/usr/bin/env python3
"""Train GatedFusion model with scaffold 10-fold CV on property-matched data."""

import os, sys, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.fusion_model import FusionModel, build_ablation
from scripts.scaffold_split import ScaffoldKFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Helpers ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.5, "aupr": 0.5, "ef_1pct": 0, "ef_5pct": 0, "ef_10pct": 0,
                "bedroc": 0, "acc": 0.5, "f1": 0.5,
                "recall": 0.5, "specificity": 0.5, "mcc": 0.0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n_pos = int(y_true.sum())
    n_total = len(y_true)

    # Enrichment Factor at top k%
    def _ef(k):
        n_top = max(1, int(n_total * k / 100))
        idx = np.argsort(y_prob)[::-1][:n_top]
        hits = y_true[idx].sum()
        return (hits / n_top) / (n_pos / n_total) if n_pos > 0 else 0.0

    # BEDROC (alpha=20, weights early recognition heavily)
    def _bedroc(alpha=20.0):
        order = np.argsort(y_prob)[::-1]
        ranks = np.where(y_true[order] == 1)[0]
        if len(ranks) == 0:
            return 0.0
        n = n_total
        ri = 1.0 - np.exp(-alpha)  # R_a(1)
        s = np.sum(np.exp(-alpha * ranks / n))
        ra = n_pos / n
        top = ra * np.sinh(alpha/2) / (np.cosh(alpha/2) - np.cosh(alpha/2 - alpha*ra)) if ra > 0 else 1.0
        bottom = ri / (1 - np.exp(-alpha))
        return s / (ra * n * ri) * ra * (1 - np.exp(-alpha)) / (1 - np.exp(-alpha*ra)) if ra > 0 else 0.0

    return {
        "auc": roc_auc_score(y_true, y_prob),
        "aupr": average_precision_score(y_true, y_prob),
        "ef_1pct": _ef(1), "ef_5pct": _ef(5), "ef_10pct": _ef(10),
        "bedroc": _bedroc(20),
        "acc": accuracy_score(y_true, y_pred),
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def find_best_f1(y_true, y_prob, beta=0.5):
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (y_prob >= t).astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0)
        fb = ((1 + beta**2) * p * r / ((beta**2)*p + r) if (beta**2*p + r) > 0
              else 0.0)
        if fb > best:
            best, best_t = fb, float(t)
    return best, best_t


# ── Training ─────────────────────────────────────────────────────

def forward_variant(model, xm, xu, variant):
    if variant == "unimol_only":
        return model(xu, xm)  # (input, _dummy)
    return model(xm, xu)


def train_epoch(model, loader, opt, criterion, device, variant):
    model.train()
    total_loss = 0.0
    for xm, xu, y in loader:
        xm, xu, y = xm.to(device), xu.to(device), y.to(device)
        opt.zero_grad()
        pred, _ = forward_variant(model, xm, xu, variant)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, variant):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xm, xu, y in loader:
            out, _ = forward_variant(model, xm.to(device), xu.to(device), variant)
            preds.extend(out.cpu().numpy().flatten())
            labels.extend(y.numpy().flatten())
    return np.array(labels), np.array(preds)


# ── Main ─────────────────────────────────────────────────────────

def run_experiment(
    X_molclr, X_unimol, y, smiles_list,
    model_variant="full", split_method="scaffold",
    embed_dim=256, dropout=0.3, lr=1e-4, epochs=100, patience=15,
    batch_size=32, n_folds=10, seed=42, output_dir="results/fusion_baseline",
):
    os.makedirs(output_dir, exist_ok=True)

    Xm_t = torch.from_numpy(X_molclr).float()
    Xu_t = torch.from_numpy(X_unimol).float()
    y_t = torch.from_numpy(y).float().view(-1, 1)

    print(f"\n{'='*60}")
    print(f"Fusion Experiment: {model_variant} [{split_method}]")
    print(f"  Data: {len(y)} samples (P={int(y.sum())}, N={int(len(y)-y.sum())})")
    print(f"  embed_dim={embed_dim}, dropout={dropout}, lr={lr}, epochs={epochs}")
    print(f"{'='*60}")

    # Split
    if split_method == "scaffold":
        kfold = ScaffoldKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(Xm_t, y_t.numpy(), smiles_list))
    else:
        from sklearn.model_selection import StratifiedKFold
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(Xm_t.numpy(), y))

    all_metrics = {"auc": [], "aupr": [], "ef_1pct": [], "ef_5pct": [],
                   "ef_10pct": [], "bedroc": [],
                   "acc": [], "f1": [], "recall": [], "specificity": [], "mcc": []}
    all_gate_alphas = []
    all_val_labels, all_val_preds = [], []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n── Fold {fold+1}/{n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        train_loader = DataLoader(
            TensorDataset(Xm_t[train_idx], Xu_t[train_idx], y_t[train_idx]),
            batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(
            TensorDataset(Xm_t[val_idx], Xu_t[val_idx], y_t[val_idx]),
            batch_size=batch_size, shuffle=False)

        # Build model
        model_kwargs = {
            "molclr_dim": 512, "unimol_dim": 768,
            "embed_dim": embed_dim, "dropout_rate": dropout,
        }
        model = build_ablation(model_variant, **model_kwargs).to(device)

        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        criterion = nn.BCELoss()

        best_auc, patience_cnt = -1.0, 0
        best_state = None

        for epoch in range(epochs):
            train_epoch(model, train_loader, opt, criterion, device, model_variant)
            scheduler.step()

            labels, preds = evaluate(model, val_loader, device, model_variant)
            if len(np.unique(labels)) < 2:
                continue
            auc = roc_auc_score(labels, preds)

            if auc > best_auc + 1e-6:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

            if (epoch + 1) % 20 == 0:
                print(f"  epoch {epoch+1:3d}  val_auc={auc:.4f}  best={best_auc:.4f}")

        model.load_state_dict(best_state)
        labels, preds = evaluate(model, val_loader, device, model_variant)
        _, best_t = find_best_f1(labels, preds, beta=1.0)
        m = compute_metrics(labels, preds, threshold=best_t)

        print(f"  Fold {fold+1} done: AUC={m['auc']:.4f} AUPR={m['aupr']:.4f} "
              f"EF1%={m['ef_1pct']:.1f} EF5%={m['ef_5pct']:.1f} "
              f"BEDROC={m['bedroc']:.3f} thresh={best_t:.3f}")

        # Record gate values
        if model_variant == "full":
            model.eval()
            with torch.no_grad():
                xm_v = Xm_t[val_idx].to(device)
                xu_v = Xu_t[val_idx].to(device)
                _, alphas = model(xm_v, xu_v)
                all_gate_alphas.append(alphas.cpu().numpy().flatten())

        for k in all_metrics:
            all_metrics[k].append(m[k])
        all_val_labels.extend(labels)
        all_val_preds.extend(preds)

    # ── Pooled results ──
    print(f"\n{'='*60}")
    print(f"Results: {model_variant}")
    for k in ["auc", "aupr", "bedroc", "ef_1pct", "ef_5pct", "ef_10pct",
              "f1", "recall", "specificity", "mcc"]:
        vals = np.array(all_metrics[k])
        print(f"  {k:12s}: {vals.mean():.4f} ± {vals.std():.4f}")

    results = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for k, v in all_metrics.items()}
    results["model_variant"] = model_variant

    # Gate analysis
    if all_gate_alphas:
        all_alphas = np.concatenate(all_gate_alphas)
        results["gate_alpha_mean"] = float(all_alphas.mean())
        results["gate_alpha_std"] = float(all_alphas.std())
        print(f"  Gate α: {all_alphas.mean():.3f} ± {all_alphas.std():.3f} "
              f"(α→1 = MolCLR dominant, α→0 = Uni-Mol2 dominant)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/fusion_baseline")
    parser.add_argument("--split", default="scaffold",
                       choices=["scaffold", "random"])
    parser.add_argument("--replicate", type=int, default=0,
                       help="Which property-matched replicate to use (0-9)")
    args = parser.parse_args()

    # 使用预对齐的数据（MolCLR dataset_pred.py 有跳过首行的 bug）
    aligned_dir = "data/processed"
    csv_path = os.path.join(aligned_dir, "aligned_data.csv")
    molclr_path = os.path.join(aligned_dir, "aligned_molclr.npy")
    unimol_path = os.path.join(aligned_dir, "aligned_unimol.npy")
    y_path = os.path.join(aligned_dir, "aligned_y.npy")

    print(f"Loading aligned data: {csv_path}")
    df = pd.read_csv(csv_path)
    y = np.load(y_path).astype(np.float32)
    smiles_list = df["smiles"].tolist()
    X_molclr = np.load(molclr_path).astype(np.float32)
    X_unimol = np.load(unimol_path).astype(np.float32)

    print(f"  MolCLR={X_molclr.shape}, UniMol={X_unimol.shape}, y={y.shape}")

    run_experiment(
        X_molclr, X_unimol, y, smiles_list,
        model_variant=args.variant,
        split_method=args.split,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        output_dir=args.output_dir,
    )
