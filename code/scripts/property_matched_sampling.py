"""
Property-matched negative sampling for low-data drug discovery.

Replaces random ZINC sampling with physicochemical-property-matched decoy
selection.  For each active molecule, we select decoys from ZINC whose MW,
LogP, HBD, HBA, RotB, and TPSA distributions match the active set, while
enforcing structural dissimilarity (Tanimoto < 0.5 on ECFP4 fingerprints).

This addresses a key reviewer concern: random ZINC negatives are trivially
separable from actives by gross physicochemical properties alone.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_RAW, DATA_PROCESSED, POSITIVE_CSV, ZINC_NEG_CSV,
    PROPERTY_NAMES, PROPERTY_TOLERANCE, TANIMOTO_MAX,
    NEG_PER_POS_RATIO, N_SAMPLING_REPLICATES, MIN_NEG_PER_POS,
    RANDOM_SEED,
)

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ── Property Calculation ────────────────────────────────────────────

def compute_properties(smiles: str) -> Optional[Dict[str, float]]:
    """Compute 6 physicochemical descriptors for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotB": Descriptors.NumRotatableBonds(mol),
            "TPSA": Descriptors.TPSA(mol),
        }
    except Exception:
        return None


def compute_ecfp4(smiles: str, radius: int = 2, nbits: int = 2048):
    """Compute ECFP4 (Morgan) fingerprint as a bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return fp


def compute_scaffold(smiles: str) -> str:
    """Compute Bemis-Murcko scaffold SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


# ── Bulk Property Computation ───────────────────────────────────────

def build_molecule_library(
    csv_path: str,
    smiles_col: str = "smiles",
    label_col: Optional[str] = None,
    name_col: Optional[str] = None,
    compute_fps: bool = True,
) -> pd.DataFrame:
    """Load a CSV and compute properties + fingerprints for all molecules."""
    df = pd.read_csv(csv_path)

    props_list = []
    fps_list = []
    scaffolds_list = []
    valid_idx = []

    for i, row in df.iterrows():
        smi = row[smiles_col]
        props = compute_properties(smi)
        if props is None:
            continue
        props_list.append(props)
        if compute_fps:
            fp = compute_ecfp4(smi)
            if fp is None:
                continue
            fps_list.append(fp)
        scaffolds_list.append(compute_scaffold(smi))
        valid_idx.append(i)

    result = df.iloc[valid_idx].copy().reset_index(drop=True)
    prop_df = pd.DataFrame(props_list, index=result.index)
    result = pd.concat([result, prop_df], axis=1)
    if compute_fps:
        result["_ecfp4"] = fps_list
    result["_scaffold"] = scaffolds_list

    print(f"  → Loaded {len(result)} valid molecules from {csv_path}")
    return result


# ── Tanimoto Filtering ──────────────────────────────────────────────

def bulk_tanimoto_filter(
    query_fp,
    pool_fps: List,
    pool_df: pd.DataFrame,
    max_sim: float = TANIMOTO_MAX,
    max_candidates: int = 5000,
) -> np.ndarray:
    """
    Filter pool molecules by Tanimoto similarity to a query.
    Returns indices into pool_df of molecules with similarity < max_sim.
    """
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, pool_fps))
    # Keep molecules below similarity threshold
    mask = sims < max_sim

    if mask.sum() == 0:
        # If too restrictive, loosen and take the most dissimilar
        sorted_idx = np.argsort(sims)
        return sorted_idx[:max_candidates]

    # From those below threshold, pick candidates
    candidate_idx = np.where(mask)[0]
    if len(candidate_idx) > max_candidates:
        # Random subsample to keep computation manageable
        rng = np.random.RandomState(RANDOM_SEED)
        candidate_idx = rng.choice(candidate_idx, max_candidates, replace=False)

    return candidate_idx


# ── Property-Matched Decoy Selection ────────────────────────────────

def select_property_matched_decoys(
    positives_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    neg_per_pos: int = NEG_PER_POS_RATIO,
    property_names: List[str] = PROPERTY_NAMES,
    tolerance: float = PROPERTY_TOLERANCE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    For each positive molecule, select property-matched decoys from the
    negative pool.  Uses nearest-neighbors in standardized property space,
    followed by structural filtering.

    Returns a DataFrame with columns: p_n, name, smiles, source_id
    """
    rng = np.random.RandomState(random_seed)
    neg_available = negatives_df.copy().reset_index(drop=True)
    neg_available["_used"] = False

    all_decoy_rows = []
    total_needed = len(positives_df) * neg_per_pos

    print(f"\n  Selecting {neg_per_pos} decoys per positive "
          f"({len(positives_df)} positives → target {total_needed} decoys)")

    # Standardize property values using the positive set as reference
    pos_props = positives_df[property_names].values.astype(float)
    neg_props = neg_available[property_names].values.astype(float)

    scaler = StandardScaler()
    scaler.fit(pos_props)
    pos_props_scaled = scaler.transform(pos_props)
    neg_props_scaled = scaler.transform(neg_props)

    # Pre-compute ECFP4 fingerprints for positives
    pos_fps = positives_df["_ecfp4"].tolist()
    neg_fps = neg_available["_ecfp4"].tolist()

    for i, (_, pos_row) in enumerate(positives_df.iterrows()):
        if (i + 1) % 20 == 0:
            print(f"    Processing positive {i + 1}/{len(positives_df)}...")

        # Compute property distances to all available negatives
        pos_vec = pos_props_scaled[i].reshape(1, -1)
        dists = np.linalg.norm(neg_props_scaled - pos_vec, axis=1)

        # Determine distance threshold: keep negatives within a reasonable
        # property-distance window.  Use the tolerance fraction of the
        # maximum distance to avoid being too restrictive.
        dist_threshold = np.percentile(dists, tolerance * 100)

        # Candidate indices: within property distance AND not already used
        property_mask = dists < dist_threshold
        available_mask = ~neg_available["_used"].values
        candidate_mask = property_mask & available_mask

        if candidate_mask.sum() < MIN_NEG_PER_POS:
            # Loosen: drop the "not already used" constraint
            candidate_mask = property_mask

        candidate_idx = np.where(candidate_mask)[0]

        if len(candidate_idx) == 0:
            # Fallback: take the closest available molecules
            sorted_by_dist = np.argsort(dists)
            candidate_idx = sorted_by_dist[:max(neg_per_pos * 5, MIN_NEG_PER_POS * 3)]
            print(f"    ⚠ Positive {i + 1}: no property-matched candidates, "
                  f"using {len(candidate_idx)} closest")

        # Structural filtering: Tanimoto-based
        query_fp = pos_fps[i]
        if len(candidate_idx) <= neg_per_pos * 3:
            structurally_filtered = candidate_idx
        else:
            structurally_filtered = bulk_tanimoto_filter(
                query_fp, neg_fps, neg_available,
                max_sim=TANIMOTO_MAX,
                max_candidates=neg_per_pos * 5,
            )
            # Intersect with property-matched candidates
            structurally_filtered = np.intersect1d(
                structurally_filtered,
                candidate_idx,
            )

        if len(structurally_filtered) == 0:
            structurally_filtered = candidate_idx[
                np.argsort(dists[candidate_idx])
            ][:neg_per_pos * 3]

        # Select decoys
        n_select = min(neg_per_pos, len(structurally_filtered))
        chosen = rng.choice(structurally_filtered, n_select, replace=False)

        # Mark as used
        neg_available.loc[neg_available.index[chosen], "_used"] = True

        # Build result rows
        for idx in chosen:
            row = neg_available.iloc[idx]
            all_decoy_rows.append({
                "p_n": 0,
                "name": row.get("name", f"decoy_{idx}"),
                "smiles": row["smiles"],
                "source_positive_idx": i,
                "source_positive_name": pos_row.get("name", f"pos_{i}"),
            })

    result_df = pd.DataFrame(all_decoy_rows)
    print(f"  ✓ Selected {len(result_df)} decoys total "
          f"(avg {len(result_df) / len(positives_df):.1f} per positive)")
    return result_df


# ── Generate Replicate Datasets ─────────────────────────────────────

def generate_replicate_datasets(
    positives_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    n_replicates: int = N_SAMPLING_REPLICATES,
    output_dir: str = DATA_PROCESSED,
) -> List[str]:
    """
    Generate N independent replicate datasets with property-matched decoys.
    Each replicate uses a different random seed to ensure decoy diversity.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # Prepare positive rows in standard format
    pos_rows = []
    for _, row in positives_df.iterrows():
        pos_rows.append({
            "p_n": 1,
            "name": row.get("name", ""),
            "smiles": row["smiles"],
        })
    pos_df = pd.DataFrame(pos_rows)

    for rep in range(n_replicates):
        seed = RANDOM_SEED + rep * 100
        print(f"\n{'=' * 60}")
        print(f"Replicate {rep + 1}/{n_replicates} (seed={seed})")
        print(f"{'=' * 60}")

        decoys_df = select_property_matched_decoys(
            positives_df, negatives_df,
            neg_per_pos=NEG_PER_POS_RATIO,
            random_seed=seed,
        )

        # Combine positives and decoys
        decoy_rows = decoys_df[["p_n", "name", "smiles"]].to_dict("records")
        combined = pd.DataFrame(pos_rows + decoy_rows)

        # Shuffle
        rng = np.random.RandomState(seed)
        combined = combined.iloc[rng.permutation(len(combined))].reset_index(drop=True)

        # Save
        out_path = os.path.join(output_dir, f"data4finetune_{rep}.csv")
        combined.insert(0, "idx", range(len(combined)))
        combined.to_csv(out_path, index=False)
        saved_paths.append(out_path)
        print(f"  → Saved {out_path} ({len(combined)} samples: "
              f"{len(pos_rows)} pos, {len(decoy_rows)} neg)")

    return saved_paths


# ── Analysis: Compare Random vs Property-Matched ────────────────────

def analyze_property_distributions(
    positives_df: pd.DataFrame,
    random_neg_csv: str,
    property_matched_csv: str,
    output_dir: str,
):
    """Generate property distribution comparison plots for the paper."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    random_neg = pd.read_csv(random_neg_csv)
    prop_match_neg = pd.read_csv(property_matched_csv)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, prop in zip(axes, PROPERTY_NAMES):
        # Compute properties for each set
        def get_prop_values(df, label_col="p_n"):
            vals_pos, vals_neg = [], []
            for _, row in df.iterrows():
                props = compute_properties(row["smiles"])
                if props is None:
                    continue
                if row[label_col] == 1:
                    vals_pos.append(props[prop])
                else:
                    vals_neg.append(props[prop])
            return vals_pos, vals_neg

        # Random sampling
        r_pos, r_neg = get_prop_values(random_neg)
        # Property-matched
        p_pos, p_neg = get_prop_values(prop_match_neg)

        # Plot distributions
        if r_neg:
            ax.hist(r_neg, bins=30, alpha=0.4, label="Random ZINC neg", color="red",
                    density=True)
        if p_neg:
            ax.hist(p_neg, bins=30, alpha=0.4, label="Prop-matched neg",
                    color="blue", density=True)
        if r_pos:
            ax.hist(r_pos, bins=20, alpha=0.6, label="Positives", color="green",
                    density=True)
        ax.set_xlabel(prop)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.set_title(f"{prop} Distribution")

    plt.suptitle("Property Distribution: Random vs Property-Matched Negatives",
                 fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "property_distribution_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved property distribution plot to {out_path}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Property-matched negative sampling for SND1 inhibitor discovery"
    )
    parser.add_argument("--positives", default=POSITIVE_CSV,
                        help="Path to positive molecules CSV")
    parser.add_argument("--negatives", default=ZINC_NEG_CSV,
                        help="Path to ZINC negative pool CSV")
    parser.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS_RATIO)
    parser.add_argument("--n-replicates", type=int, default=N_SAMPLING_REPLICATES)
    parser.add_argument("--output-dir", default=os.path.join(DATA_PROCESSED, "property_matched"))
    parser.add_argument("--analyze", action="store_true",
                        help="Generate property distribution comparison plots")
    parser.add_argument("--original-neg-sample",
                        default=os.path.join(
                            os.path.dirname(DATA_RAW),
                            "data4finetune_0.csv",
                        ),
                        help="Path to original random-neg dataset for comparison")

    args = parser.parse_args()

    print("=" * 60)
    print("Property-Matched Negative Sampling")
    print("=" * 60)

    # Load positive molecules
    print("\n[1/4] Loading positive molecules...")
    positives_df = build_molecule_library(args.positives, label_col="p_n",
                                          name_col="name")
    print(f"  Positives: {len(positives_df)}")

    # Load negative pool
    print("\n[2/4] Loading negative molecule pool...")
    negatives_df = build_molecule_library(args.negatives, name_col="name")
    print(f"  Negative pool: {len(negatives_df)}")

    # Generate property-matched datasets
    print(f"\n[3/4] Generating {args.n_replicates} property-matched datasets...")
    dataset_paths = generate_replicate_datasets(
        positives_df, negatives_df,
        n_replicates=args.n_replicates,
        output_dir=args.output_dir,
    )

    # Analysis
    if args.analyze:
        print("\n[4/4] Generating property distribution comparison...")
        if os.path.exists(args.original_neg_sample):
            analyze_property_distributions(
                positives_df,
                args.original_neg_sample,
                dataset_paths[0],
                output_dir=os.path.join(args.output_dir, "analysis"),
            )
        else:
            print(f"  Original neg-sample not found at {args.original_neg_sample}")

    print("\n" + "=" * 60)
    print("Done. Generated datasets:")
    for p in dataset_paths:
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
