#!/usr/bin/env python3
"""
Batch extract Uni-Mol2 1.1B features for 10 property-matched datasets.

Uses unimol_tools.UniMolRepr (no unicore needed).

Output: (n_molecules, 1536) per replicate
"""

import os, sys
import numpy as np
import pandas as pd

from unimol_tools.predictor import UniMolRepr
import torch


def main():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data/processed/property_matched"
    )
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading Uni-Mol2 1.1B model...")
    model = UniMolRepr(model_name="unimolv2", model_size="1.1B", use_cuda=(device == "cuda"))

    for rep in range(10):
        csv_path = os.path.join(data_dir, f"data4finetune_{rep}.csv")
        out_path = os.path.join(out_dir, f"unimol_vec_{rep}.npy")

        if os.path.exists(out_path):
            print(f"[{rep}] Already exists, skipping")
            continue

        print(f"[{rep}] Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        smiles_list = df["smiles"].tolist()
        print(f"  {len(smiles_list)} molecules")

        vectors = model.get_repr(smiles_list, return_tensor=True)
        print(f"  → {vectors.shape}")

        np.save(out_path, vectors.cpu().numpy())
        print(f"  Saved to {out_path}")

    print("All done!")


if __name__ == "__main__":
    main()
