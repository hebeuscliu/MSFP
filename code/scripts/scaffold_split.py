"""
Scaffold-based data splitting for rigorous model validation.

Implements Bemis-Murcko scaffold split to ensure molecules sharing the same
core scaffold never appear in both training and validation sets.  This is a
much harder test than random split and directly addresses the reviewer's
concern about "random 10-fold cross-validation does not demonstrate scaffold
or prospective generalization."

Also provides a comparison framework: random split vs scaffold split on the
same data, so the paper can report the generalization gap.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Optional, Iterator, Dict
from sklearn.model_selection import StratifiedKFold, KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ── Scaffold K-Fold ─────────────────────────────────────────────────

class ScaffoldKFold:
    """
    Bemis-Murcko scaffold-based K-Fold splitter.

    Molecules with the same scaffold are always assigned to the same fold,
    ensuring no scaffold-level data leakage between training and validation.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 10).
    shuffle : bool
        Whether to shuffle scaffold order before assignment.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, n_splits: int = 10, shuffle: bool = True,
                 random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _generate_scaffold(self, smiles: str,
                           include_chirality: bool = False) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=include_chirality
            )
        except Exception:
            return ""

    def split(self, X, y, smiles_list: List[str]) -> Iterator[
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Generate train/val index splits based on scaffold grouping.

        Yields (train_indices, val_indices) for each fold.
        """
        print(f"  Computing scaffolds for {len(smiles_list)} molecules...")

        scaffold_to_indices = defaultdict(list)
        for idx, smi in enumerate(smiles_list):
            scaffold = self._generate_scaffold(str(smi))
            scaffold_to_indices[scaffold].append(idx)

        scaffolds = list(scaffold_to_indices.keys())
        n_unique_scaffolds = len(scaffolds)
        print(f"  Found {n_unique_scaffolds} unique scaffolds")

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(scaffolds)
        else:
            # Sort by scaffold size (largest first) for balanced folds
            scaffolds.sort(
                key=lambda s: len(scaffold_to_indices[s]), reverse=True
            )

        # Distribute scaffolds across folds (greedy minimization of size imbalance)
        folds = [[] for _ in range(self.n_splits)]
        fold_sizes = [0] * self.n_splits

        for scaff in scaffolds:
            idxs = scaffold_to_indices[scaff]
            min_fold = fold_sizes.index(min(fold_sizes))
            folds[min_fold].extend(idxs)
            fold_sizes[min_fold] += len(idxs)

        print(f"  Fold sizes: {fold_sizes}")

        for i in range(self.n_splits):
            val_indices = np.array(folds[i])
            train_indices = np.concatenate([
                folds[j] for j in range(self.n_splits) if j != i
            ])

            if self.shuffle:
                rng = np.random.RandomState(self.random_state + i)
                rng.shuffle(train_indices)

            yield train_indices, val_indices


# ── Split Comparison Tool ───────────────────────────────────────────

def compare_split_strategies(
    csv_path: str,
    smiles_col: str = "smiles",
    label_col: str = "p_n",
    n_splits: int = 10,
    random_state: int = 42,
) -> Dict:
    """
    Compare random stratified split vs scaffold split on the same dataset.

    Returns a dict with overlap statistics between the two strategies,
    useful for demonstrating why scaffold split is harder.
    """
    df = pd.read_csv(csv_path)
    X = np.arange(len(df))
    y = df[label_col].values
    smiles_list = df[smiles_col].tolist()

    # Random stratified split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    random_splits = list(skf.split(X, y))

    # Scaffold split
    scaffold_kfold = ScaffoldKFold(n_splits=n_splits, shuffle=True,
                                   random_state=random_state)
    scaffold_splits = list(scaffold_kfold.split(X, y, smiles_list))

    # Analysis
    results = {
        "n_splits": n_splits,
        "n_samples": len(df),
        "n_positives": int(y.sum()),
        "n_negatives": int(len(y) - y.sum()),
    }

    # Count scaffolds
    scaffolds = [ScaffoldKFold()._generate_scaffold(s) for s in smiles_list]
    scaffold_counts = defaultdict(int)
    for s in scaffolds:
        scaffold_counts[s] += 1
    results["n_unique_scaffolds"] = len(scaffold_counts)
    results["n_multi_mol_scaffolds"] = sum(
        1 for c in scaffold_counts.values() if c > 1
    )

    # Compute train/val scaffold overlap for random split
    random_overlap_rates = []
    for fold, (train_idx, val_idx) in enumerate(random_splits):
        train_scaffolds = set(scaffolds[i] for i in train_idx)
        val_scaffolds = set(scaffolds[i] for i in val_idx)
        overlap = len(train_scaffolds & val_scaffolds)
        overlap_frac = overlap / len(val_scaffolds) if val_scaffolds else 0
        random_overlap_rates.append(overlap_frac)

    results["random_split_avg_scaffold_overlap"] = np.mean(random_overlap_rates)

    # Scaffold split has 0 overlap by construction
    results["scaffold_split_avg_scaffold_overlap"] = 0.0

    # Print summary
    print("\n" + "=" * 60)
    print("Split Strategy Comparison")
    print("=" * 60)
    print(f"  Samples: {results['n_samples']} "
          f"(P={results['n_positives']}, N={results['n_negatives']})")
    print(f"  Unique scaffolds: {results['n_unique_scaffolds']}")
    print(f"  Multi-molecule scaffolds: {results['n_multi_mol_scaffolds']}")
    print(f"  Random split scaffold overlap: "
          f"{results['random_split_avg_scaffold_overlap']:.3f}")
    print(f"  Scaffold split scaffold overlap: 0.0 (by design)")
    print("=" * 60)

    return results


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scaffold-based data splitting utilities"
    )
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to CSV to compare random vs scaffold split")
    parser.add_argument("--splits", type=int, default=10)

    args = parser.parse_args()

    if args.compare:
        compare_split_strategies(args.compare, n_splits=args.splits)
    else:
        print("Usage: python scaffold_split.py --compare <dataset.csv>")
        print("\nExample:")
        print("  python scaffold_split.py --compare data/processed/data4finetune_0.csv")
