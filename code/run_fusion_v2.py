#!/usr/bin/env python3
"""
Improved fusion training with SE-Block, contrastive loss, and Mixup.

Pipeline:
  1. Train on 10 property-matched replicates → save 10 model checkpoints
  2. Each replicate: internal scaffold 80/20 split, best-val-AUC saved
  3. Ensemble soft-voting on LMD screening

Usage:
  python run_fusion_v2.py --train        # train 10 models
  python run_fusion_v2.py --screen       # score LMD with ensemble
"""

import os, sys, json, argparse, random, glob
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
from models.fusion_model import (
    FusionModel, build_ablation,
    contrastive_alignment_loss, mixup_features,
)
from scripts.scaffold_split import ScaffoldKFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJ_ROOT, "data", "processed", "property_matched_v2")
MODEL_DIR = os.path.join(PROJ_ROOT, "models", "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.5, "aupr": 0.5, "ef_1pct": 0, "ef_5pct": 0,
                "ef_10pct": 0, "bedroc": 0, "acc": 0.5, "f1": 0.5,
                "recall": 0.5, "specificity": 0.5, "mcc": 0.0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n_pos = int(y_true.sum())
    n_total = len(y_true)

    def _ef(k):
        n_top = max(1, int(n_total * k / 100))
        idx = np.argsort(y_prob)[::-1][:n_top]
        hits = y_true[idx].sum()
        return (hits / n_top) / (n_pos / n_total) if n_pos > 0 else 0.0

    def _bedroc(alpha=20.0):
        order = np.argsort(y_prob)[::-1]
        ranks = np.where(y_true[order] == 1)[0]
        if len(ranks) == 0:
            return 0.0
        n = n_total
        ra = n_pos / n
        ri = 1.0 - np.exp(-alpha)
        s = np.sum(np.exp(-alpha * ranks / n))
        return s / (ra * n * ri) * ra * (1 - np.exp(-alpha)) / (1 - np.exp(-alpha * ra)) if ra > 0 else 0.0

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


def forward_variant(model, xm, xu, variant):
    if variant in ("unimol_only",):
        return model(xu, xm)
    return model(xm, xu)


# ── Training ─────────────────────────────────────────────────────

def train_epoch(model, loader, opt, criterion, device, variant,
                lambda_contrastive=0.02, mixup_alpha=0.4):
    model.train()
    total_loss, total_bce, total_contr = 0.0, 0.0, 0.0
    for xm, xu, y in loader:
        xm, xu, y = xm.to(device), xu.to(device), y.to(device)

        # Mixup (50% probability)
        if mixup_alpha > 0 and random.random() < 0.5:
            xm, xu, y = mixup_features(xm, xu, y, mixup_alpha)

        opt.zero_grad()

        result = forward_variant(model, xm, xu, variant)
        if len(result) == 4:  # return_embeds=True
            pred, gate, m_emb, u_emb = result
            contr_loss = contrastive_alignment_loss(m_emb, u_emb)
        else:
            pred, gate = result
            contr_loss = torch.tensor(0.0, device=device)

        bce_loss = criterion(pred, y)
        loss = bce_loss + lambda_contrastive * contr_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_contr += contr_loss.item() if isinstance(contr_loss, float) else contr_loss.item()
    n = len(loader)
    return total_loss / n, total_bce / n, total_contr / n


@torch.no_grad()
def evaluate(model, loader, device, variant):
    model.eval()
    preds, labels = [], []
    for xm, xu, y in loader:
        out, _ = forward_variant(model, xm.to(device), xu.to(device), variant)
        preds.extend(out.cpu().numpy().flatten())
        labels.extend(y.numpy().flatten())
    return np.array(labels), np.array(preds)


def train_one_replicate(
    X_molclr, X_unimol, y, smiles_list,
    replicate_id=0, model_variant="full",
    embed_dim=256, dropout=0.2, lr=1e-4, epochs=150, patience=20,
    batch_size=32, seed=42,
    lambda_contrastive=0.02, mixup_alpha=0.4,
    fusion_type="se_block",
):
    """Train one model on one replicate, return best AUC and save checkpoint."""
    set_seed(seed + replicate_id)

    Xm_t = torch.from_numpy(X_molclr).float()
    Xu_t = torch.from_numpy(X_unimol).float()
    y_t = torch.from_numpy(y).float().view(-1, 1)

    print(f"\n{'='*60}")
    print(f"Replicate {replicate_id} | {model_variant} (fusion={fusion_type})")
    print(f"  Data: {len(y)} (P={int(y.sum())}, N={int(len(y)-y.sum())})")
    print(f"  λ_contr={lambda_contrastive}, mixup_α={mixup_alpha}")
    print(f"{'='*60}")

    # Scaffold split (80/20)
    kfold = ScaffoldKFold(n_splits=5, shuffle=True, random_state=seed + replicate_id)
    splits = list(kfold.split(Xm_t, y_t.numpy(), smiles_list))
    train_idx, val_idx = splits[0]  # use first fold split

    train_loader = DataLoader(
        TensorDataset(Xm_t[train_idx], Xu_t[train_idx], y_t[train_idx]),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(Xm_t[val_idx], Xu_t[val_idx], y_t[val_idx]),
        batch_size=batch_size, shuffle=False)

    # Build model
    model_kwargs = {
        "molclr_dim": X_molclr.shape[1],
        "unimol_dim": X_unimol.shape[1],
        "embed_dim": embed_dim,
        "dropout_rate": dropout,
        "fusion_type": fusion_type,
    }
    model = build_ablation(model_variant, **model_kwargs).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # BCE with pos_weight for class imbalance
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    criterion = nn.BCELoss()  # pos_weight not needed with mixup (labels become soft)

    best_auc, patience_cnt = -1.0, 0
    best_state = None

    for epoch in range(epochs):
        train_loss, bce, contr = train_epoch(
            model, train_loader, opt, criterion, device, model_variant,
            lambda_contrastive=lambda_contrastive, mixup_alpha=mixup_alpha,
        )
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
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d} | loss={train_loss:.4f} bce={bce:.4f} "
                  f"contr={contr:.4f} | val_auc={auc:.4f} best={best_auc:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(
        MODEL_DIR, f"fusion_{model_variant}_rep{replicate_id}.pth")
    torch.save({
        "model_state_dict": best_state,
        "val_auc": best_auc,
        "replicate_id": replicate_id,
        "model_kwargs": model_kwargs,
        "model_variant": model_variant,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path} (val_auc={best_auc:.4f})")

    # Final evaluation on validation set
    model.load_state_dict(best_state)
    labels, preds = evaluate(model, val_loader, device, model_variant)
    metrics = compute_metrics(labels, preds)
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ Rep {replicate_id:2d}  AUC:   {metrics['auc']:.4f}                       │")
    print(f"  │       AUPR:  {metrics['aupr']:.4f}                       │")
    print(f"  │       F1:    {metrics['f1']:.4f}                       │")
    print(f"  │       MCC:   {metrics['mcc']:.4f}                       │")
    print(f"  │       EF1%:  {metrics['ef_1pct']:.1f}                       │")
    print(f"  │       EF5%:  {metrics['ef_5pct']:.1f}                       │")
    print(f"  │       EF10%: {metrics['ef_10pct']:.1f}                       │")
    print(f"  │       BEDROC:{metrics['bedroc']:.3f}                       │")
    print(f"  │       ACC:   {metrics['acc']:.4f}                       │")
    print(f"  │       Recall:{metrics['recall']:.4f}                       │")
    print(f"  │       Spec:  {metrics['specificity']:.4f}                       │")
    print(f"  └─────────────────────────────────────────────┘")

    return best_auc, metrics


# ── Main: Training ──────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_all_replicates(args):
    """Train on all 10 property-matched replicates."""
    all_results = []

    for rep in range(10):
        csv_path = os.path.join(DATA_DIR, f"data4finetune_{rep}.csv")
        df = pd.read_csv(csv_path)
        smiles_list = df["smiles"].tolist()
        y = df["p_n"].values.astype(np.float32)

        # Load pre-extracted features (extracted separately)
        molclr_path = os.path.join(DATA_DIR, f"features_molclr_rep{rep}.npy")
        unimol_path = os.path.join(DATA_DIR, f"features_unimol_rep{rep}.npy")

        if not os.path.exists(molclr_path) or not os.path.exists(unimol_path):
            print(f"[SKIP] Rep {rep}: features not found. Run feature extraction first.")
            continue

        X_molclr = np.load(molclr_path).astype(np.float32)
        X_unimol = np.load(unimol_path).astype(np.float32)

        best_auc, metrics = train_one_replicate(
            X_molclr, X_unimol, y, smiles_list,
            replicate_id=rep,
            model_variant=args.variant,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
            lambda_contrastive=args.lambda_contrastive,
            mixup_alpha=args.mixup_alpha,
            fusion_type=args.fusion_type,
        )
        all_results.append({"rep": rep, "val_auc": best_auc, "metrics": metrics})

    # Summary: all metrics with mean ± std
    metric_keys = ["auc", "aupr", "f1", "mcc", "acc", "recall", "specificity",
                   "ef_1pct", "ef_5pct", "ef_10pct", "bedroc"]
    print(f"\n{'='*70}")
    print(f"  10-Replicate Summary ({args.variant}, {args.fusion_type})")
    print(f"{'='*70}")
    print(f"  {'Metric':12s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    summary = {}
    for k in metric_keys:
        vals = [r["metrics"][k] for r in all_results]
        mean_v, std_v, min_v, max_v = np.mean(vals), np.std(vals), np.min(vals), np.max(vals)
        summary[k] = {"mean": float(mean_v), "std": float(std_v),
                      "min": float(min_v), "max": float(max_v)}
        print(f"  {k:12s}  {mean_v:8.4f}  {std_v:8.4f}  {min_v:8.4f}  {max_v:8.4f}")

    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    aucs = [r["val_auc"] for r in all_results]
    print(f"  Val AUC per rep: {[f'{a:.4f}' for a in aucs]}")
    print(f"{'='*70}")

    # Save summary to JSON
    summary["val_auc_per_rep"] = [float(a) for a in aucs]
    summary["model_variant"] = args.variant
    summary["fusion_type"] = args.fusion_type
    summary["embed_dim"] = args.embed_dim
    summary["dropout"] = args.dropout
    summary["lambda_contrastive"] = args.lambda_contrastive
    summary["mixup_alpha"] = args.mixup_alpha

    os.makedirs("results", exist_ok=True)
    import json
    json_path = f"results/summary_{args.variant}_{args.fusion_type}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")


# ── Main: LMD Screening ─────────────────────────────────────────

def screen_lmd(args):
    """Score LMD library with ensemble of 10 trained models."""
    from rdkit import Chem

    screening_dir = os.path.join(PROJ_ROOT, "data", "screening")
    lmd_csv = os.path.join(screening_dir, "LMD_clean.csv")

    print(f"Loading LMD library: {lmd_csv}")
    df_lmd = pd.read_csv(lmd_csv)

    # Load MolCLR features for LMD
    molclr_path = os.path.join(screening_dir, "LMD_molclr.npy")
    X_molclr_lmd = np.load(molclr_path).astype(np.float32)
    print(f"  MolCLR: {X_molclr_lmd.shape}")

    # Load Uni-Mol2 features for LMD (must match training dims)
    unimol_path = os.path.join(screening_dir, "LMD_unimol_570m_ft.npy")
    if not os.path.exists(unimol_path):
        unimol_path = os.path.join(screening_dir, "LMD_unimol.npy")
    X_unimol_lmd = np.load(unimol_path).astype(np.float32)
    print(f"  Uni-Mol2: {X_unimol_lmd.shape}")

    # Align: MolCLR (26685) vs Uni-Mol2 (let's check)
    n = min(len(X_molclr_lmd), len(X_unimol_lmd))
    X_molclr_lmd = X_molclr_lmd[:n]
    X_unimol_lmd = X_unimol_lmd[:n]
    df_lmd = df_lmd.iloc[:n]
    print(f"  Aligned: {n} molecules")

    # Load 10 models
    ckpt_paths = sorted(glob.glob(os.path.join(MODEL_DIR, f"fusion_{args.variant}_rep*.pth")))
    if len(ckpt_paths) == 0:
        print("No trained models found. Run --train first.")
        return
    print(f"\nLoading {len(ckpt_paths)} models for ensemble...")

    all_preds = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_ablation(ckpt["model_variant"], **ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Batch predict
        Xm_t = torch.from_numpy(X_molclr_lmd).float()
        Xu_t = torch.from_numpy(X_unimol_lmd).float()
        loader = DataLoader(TensorDataset(Xm_t, Xu_t), batch_size=256, shuffle=False)

        preds = []
        with torch.no_grad():
            for (xm, xu) in loader:
                out, _ = forward_variant(model, xm.to(device), xu.to(device), ckpt["model_variant"])
                preds.extend(out.cpu().numpy().flatten())
        all_preds.append(np.array(preds))
        print(f"  {os.path.basename(ckpt_path)}: mean_pred={np.mean(preds):.4f}")

    # Ensemble: soft voting (mean)
    ensemble_preds = np.mean(all_preds, axis=0)
    df_lmd["score"] = ensemble_preds
    df_lmd = df_lmd.sort_values("score", ascending=False)

    # Output
    output_dir = os.path.join(PROJ_ROOT, "results", "screening")
    os.makedirs(output_dir, exist_ok=True)

    # Full ranked list
    df_lmd.to_csv(os.path.join(output_dir, "LMD_ranked.csv"), index=False)

    # Top-50 for docking
    top50 = df_lmd.head(50)
    top50.to_csv(os.path.join(output_dir, "LMD_top50.csv"), index=False)

    # Random-50 for baseline comparison
    rng = np.random.RandomState(42)
    rand_idx = rng.choice(len(df_lmd), 50, replace=False)
    rand50 = df_lmd.iloc[rand_idx].copy()
    rand50.to_csv(os.path.join(output_dir, "LMD_random50.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Screening results saved to {output_dir}/")
    print(f"  LMD_ranked.csv: {len(df_lmd)} compounds")
    print(f"  LMD_top50.csv: top 50 (score range: {top50['score'].iloc[0]:.4f} - {top50['score'].iloc[-1]:.4f})")
    print(f"  LMD_random50.csv: random 50 (score range: {rand50['score'].min():.4f} - {rand50['score'].max():.4f})")


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train 10 models on replicates")
    parser.add_argument("--screen", action="store_true", help="Screen LMD library with ensemble")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--fusion-type", default="se_block", choices=["se_block", "scalar_gate"])
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-contrastive", type=float, default=0.02)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    args = parser.parse_args()

    if args.train:
        train_all_replicates(args)
    elif args.screen:
        screen_lmd(args)
    else:
        print("Usage: python run_fusion_v2.py --train | --screen")
