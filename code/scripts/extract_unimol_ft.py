#!/usr/bin/env python3
"""
Extract Uni-Mol2 570M (fine-tuned) features for replicates + LMD.
Usage: python extract_unimol_ft.py <GPU_ID> <START_REP> <END_REP> [--lmd]
"""

import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from unimol_tools.predictor import UniMolRepr

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data", "processed", "property_matched_v2")
CKPT_PATH = os.path.join(PROJ_ROOT, "features", "unimol", "finetune_570m_v2", "model_0.pth")

GPU_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
START_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
END_REP = int(sys.argv[3]) if len(sys.argv) > 3 else 10
DO_LMD = "--lmd" in sys.argv
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)

print(f"Loading fine-tuned Uni-Mol2 570M from {CKPT_PATH}")
model = UniMolRepr(
    model_name="unimolv2", model_size="570m", use_cuda=True,
    pretrained_model_path=CKPT_PATH,
)

# ── Replicates ──────────────────────────────────────────────────
for rep in range(START_REP, END_REP):
    out_path = os.path.join(DATA_DIR, f"features_unimol_rep{rep}.npy")
    if os.path.exists(out_path):
        print(f"[{rep}] Already exists, skipping.")
        continue

    csv_path = os.path.join(DATA_DIR, f"data4finetune_{rep}.csv")
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].tolist()

    valid = [(i, s) for i, s in enumerate(smiles) if Chem.MolFromSmiles(s) is not None]
    valid_smiles = [s for _, s in valid]
    print(f"[{rep}] {len(valid_smiles)}/{len(smiles)} valid SMILES")

    vecs = model.get_repr(valid_smiles, return_tensor=True)
    np.save(out_path, vecs.cpu().numpy().astype(np.float32))
    print(f"[{rep}] Saved: {vecs.shape} → {out_path}")

# ── LMD Library (only if --lmd) ─────────────────────────────────
if DO_LMD:
    lmd_csv = os.path.join(PROJ_ROOT, "data", "screening", "LMD_clean.csv")
    out_lmd = os.path.join(PROJ_ROOT, "data", "screening", "LMD_unimol_570m_ft.npy")

    if not os.path.exists(out_lmd):
        df = pd.read_csv(lmd_csv)
        smiles = df["smiles"].tolist()
        valid_smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
        print(f"LMD: {len(valid_smiles)}/{len(smiles)} valid SMILES")

        vecs = model.get_repr(valid_smiles, return_tensor=True)
        np.save(out_lmd, vecs.cpu().numpy().astype(np.float32))
        print(f"LMD Saved: {vecs.shape} → {out_lmd}")
    else:
        print("LMD features already exist.")

print("\nDone.")
