"""
Analysis and visualization utilities for MSFP-MD.

Generates publication-quality figures:
  1. Property distribution comparison (random vs property-matched negatives)
  2. Random split vs scaffold split comparison (AUC gap visualization)
  3. Ablation study bar plot
  4. Confusion matrix heatmaps
  5. Scatter plot of meta-learned modality weights
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ROOT, RESULTS_DIR, PROPERTY_NAMES, DATA_PROCESSED

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ── 1. Split Strategy Comparison ────────────────────────────────────

def plot_split_strategy_comparison(
    random_results: Dict,
    scaffold_results: Dict,
    output_path: str,
):
    """Bar plot comparing random vs scaffold split performance."""
    metrics = ["auc", "aupr", "f1", "recall", "specificity", "mcc"]
    labels = ["AUC", "AUPR", "F1", "Recall", "Specificity", "MCC"]

    x = np.arange(len(metrics))
    width = 0.35

    random_means = [random_results[f"{m}_mean"] for m in metrics]
    random_stds = [random_results[f"{m}_std"] for m in metrics]
    scaffold_means = [scaffold_results[f"{m}_mean"] for m in metrics]
    scaffold_stds = [scaffold_results[f"{m}_std"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, random_means, width, yerr=random_stds,
                   label="Random 10-Fold CV", color="#4C72B0", capsize=3)
    bars2 = ax.bar(x + width/2, scaffold_means, width, yerr=scaffold_stds,
                   label="Scaffold 10-Fold CV", color="#DD8452", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Random vs Scaffold Split Comparison")
    ax.legend(loc="lower right")

    # Annotate gap
    for i, (r, s) in enumerate(zip(random_means, scaffold_means)):
        gap = r - s
        if gap > 0.02:
            ax.annotate(f"Δ={gap:.3f}", xy=(x[i], max(r, s) + 0.03),
                       ha="center", fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved split comparison to {output_path}")


# ── 2. Ablation Study Plot ──────────────────────────────────────────

def plot_ablation_study(results_dir: str, output_path: str):
    """Plot ablation results from JSON files."""
    ablation_dir = os.path.join(results_dir)
    variants = []
    data = defaultdict(list)

    for fname in sorted(os.listdir(ablation_dir)):
        if fname.startswith("results_scaffold_") and fname.endswith(".json"):
            variant = fname.replace("results_scaffold_", "").replace(".json", "")
            with open(os.path.join(ablation_dir, fname)) as f:
                r = json.load(f)

            variants.append(variant)
            for m in ["auc", "aupr", "f1", "recall", "specificity", "mcc"]:
                data[m].append(r.get(f"{m}_mean", 0))

    if not variants:
        print("  No ablation results found.")
        return

    x = np.arange(len(variants))
    width = 0.15
    metrics = ["auc", "aupr", "f1", "recall", "specificity", "mcc"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, data[m], width, label=m.upper(), color=c, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Component Contribution (Scaffold Split)")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved ablation plot to {output_path}")


# ── 3. Meta-Weight Evolution ────────────────────────────────────────

def plot_meta_weights(results_dir: str, output_path: str):
    """Plot the learned modality weights across folds."""
    scaffold_file = None
    for fname in sorted(os.listdir(results_dir)):
        if fname == "results_scaffold_full.json":
            scaffold_file = os.path.join(results_dir, fname)
            break

    if scaffold_file is None:
        print("  No full model results found for meta-weight plot.")
        return

    with open(scaffold_file) as f:
        r = json.load(f)

    # We only have mean weights, not per-fold; use per-fold metrics as proxy
    if "W_molclr_mean" not in r:
        print("  No meta-weight data in results.")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    modalities = ["MolCLR", "Uni-Mol2"]
    weights = [r.get("W_molclr_mean", 1.0), r.get("W_unimol_mean", 1.0)]
    bars = ax.bar(modalities, weights, color=["#4C72B0", "#DD8452"], alpha=0.85)
    ax.set_ylabel("Learned Weight")
    ax.set_title("Meta-Learned Modality Weights")

    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{w:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved meta-weight plot to {output_path}")


# ── 4. Per-Fold Consistency ─────────────────────────────────────────

def plot_fold_consistency(results_dir: str, output_path: str):
    """Plot per-fold AUC variability for random vs scaffold split."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for method, color, marker in [
        ("scaffold", "#DD8452", "s"),
        ("random", "#4C72B0", "o"),
    ]:
        fname = os.path.join(results_dir, f"results_{method}_full.json")
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            r = json.load(f)

        aucs = r.get("auc_per_fold", [])
        if aucs:
            ax.plot(range(1, len(aucs)+1), aucs, marker=marker, color=color,
                   label=f"{method.capitalize()} (σ={np.std(aucs):.3f})",
                   linewidth=2, markersize=6)

    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title("Per-Fold AUC Consistency")
    ax.legend()
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved fold consistency plot to {output_path}")


# ── 5. Positive-Negative Property Overlap ───────────────────────────

def plot_property_overlap(
    positives_csv: str,
    negatives_csv: str,
    output_path: str,
):
    """KDE plots showing property distribution overlap between sets."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    def load_and_compute(csv_path):
        df = pd.read_csv(csv_path)
        props = {p: [] for p in PROPERTY_NAMES}
        for smi in df["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                props["MW"].append(Descriptors.MolWt(mol))
                props["LogP"].append(Descriptors.MolLogP(mol))
                props["HBD"].append(Descriptors.NumHDonors(mol))
                props["HBA"].append(Descriptors.NumHAcceptors(mol))
                props["RotB"].append(Descriptors.NumRotatableBonds(mol))
                props["TPSA"].append(Descriptors.TPSA(mol))
            except Exception:
                continue
        return {k: np.array(v) for k, v in props.items()}

    pos_props = load_and_compute(positives_csv)
    neg_props = load_and_compute(negatives_csv)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, prop in zip(axes, PROPERTY_NAMES):
        if len(pos_props[prop]) > 0:
            ax.hist(pos_props[prop], bins=30, alpha=0.5, density=True,
                   label="Positives", color="green")
        if len(neg_props[prop]) > 0:
            ax.hist(neg_props[prop], bins=30, alpha=0.4, density=True,
                   label="Negatives", color="red")
        ax.set_xlabel(prop)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(prop)

        # Compute overlapping area metric (simplified)
        if len(pos_props[prop]) > 0 and len(neg_props[prop]) > 0:
            all_vals = np.concatenate([pos_props[prop], neg_props[prop]])
            bins = np.histogram_bin_edges(all_vals, bins=30)
            p_hist, _ = np.histogram(pos_props[prop], bins=bins, density=True)
            n_hist, _ = np.histogram(neg_props[prop], bins=bins, density=True)
            overlap = np.sum(np.minimum(p_hist, n_hist)) * (bins[1] - bins[0])
            ax.set_title(f"{prop} (overlap={overlap:.2f})", fontsize=10)

    plt.suptitle("Property Distribution: Positives vs Negatives", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → Saved property overlap plot to {output_path}")


# ── 6. Scaffold Diversity Analysis ──────────────────────────────────

def analyze_scaffold_diversity(
    csv_path: str,
    smiles_col: str = "smiles",
    label_col: str = "p_n",
    output_path: Optional[str] = None,
) -> Dict:
    """Analyze scaffold diversity: unique scaffolds, singletons, etc."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    df = pd.read_csv(csv_path)
    scaffolds_pos = defaultdict(list)
    scaffolds_neg = defaultdict(list)

    for _, row in df.iterrows():
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaff = ""
        if row[label_col] == 1:
            scaffolds_pos[scaff].append(smi)
        else:
            scaffolds_neg[scaff].append(smi)

    n_pos_scaffolds = len(scaffolds_pos)
    n_neg_scaffolds = len(scaffolds_neg)
    n_pos_singletons = sum(1 for v in scaffolds_pos.values() if len(v) == 1)
    n_neg_singletons = sum(1 for v in scaffolds_neg.values() if len(v) == 1)

    # Cross-set scaffold overlap
    pos_scaff_set = set(scaffolds_pos.keys())
    neg_scaff_set = set(scaffolds_neg.keys())
    shared = pos_scaff_set & neg_scaff_set

    results = {
        "n_positives": sum(len(v) for v in scaffolds_pos.values()),
        "n_negatives": sum(len(v) for v in scaffolds_neg.values()),
        "n_pos_scaffolds": n_pos_scaffolds,
        "n_neg_scaffolds": n_neg_scaffolds,
        "n_pos_singletons": n_pos_singletons,
        "n_neg_singletons": n_neg_singletons,
        "pos_singleton_rate": n_pos_singletons / max(n_pos_scaffolds, 1),
        "n_shared_scaffolds": len(shared),
        "shared_scaffolds": list(shared)[:10],
    }

    print(f"\nScaffold Diversity Analysis")
    print(f"  Positives: {results['n_positives']} molecules, "
          f"{n_pos_scaffolds} scaffolds "
          f"({results['pos_singleton_rate']:.1%} singleton)")
    print(f"  Negatives: {results['n_negatives']} molecules, "
          f"{n_neg_scaffolds} scaffolds")
    print(f"  Shared scaffolds: {len(shared)}")
    if len(shared) > 0:
        print(f"  ⚠ Warning: {len(shared)} scaffolds appear in BOTH sets!")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  → Saved to {output_path}")

    return results


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MSFP-MD Analysis and Visualization Tools"
    )
    sub = parser.add_subparsers(dest="command")

    # split-compare
    p = sub.add_parser("split-compare")
    p.add_argument("--random-results", required=True)
    p.add_argument("--scaffold-results", required=True)
    p.add_argument("--output", default="split_comparison.png")

    # ablation
    p = sub.add_parser("ablation")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output", default="ablation_study.png")

    # meta-weights
    p = sub.add_parser("meta-weights")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output", default="meta_weights.png")

    # fold-consistency
    p = sub.add_parser("fold-consistency")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output", default="fold_consistency.png")

    # property-overlap
    p = sub.add_parser("property-overlap")
    p.add_argument("--positives", required=True)
    p.add_argument("--negatives", required=True)
    p.add_argument("--output", default="property_overlap.png")

    # scaffold-diversity
    p = sub.add_parser("scaffold-diversity")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "split-compare":
        with open(args.random_results) as f:
            rr = json.load(f)
        with open(args.scaffold_results) as f:
            sr = json.load(f)
        plot_split_strategy_comparison(rr, sr, args.output)

    elif args.command == "ablation":
        plot_ablation_study(args.results_dir, args.output)

    elif args.command == "meta-weights":
        plot_meta_weights(args.results_dir, args.output)

    elif args.command == "fold-consistency":
        plot_fold_consistency(args.results_dir, args.output)

    elif args.command == "property-overlap":
        plot_property_overlap(args.positives, args.negatives, args.output)

    elif args.command == "scaffold-diversity":
        analyze_scaffold_diversity(args.csv, output_path=args.output)

    else:
        parser.print_help()
