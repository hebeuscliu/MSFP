#!/usr/bin/env python3
"""
Uni-Mol2 570M fine-tuning on SND1 property-matched data (v2).

Fixes from v1 (collapsed predictions AUC=0.608):
  - task='multilabel_classification' → uses sigmoid + FocalLoss
  - More epochs (50), larger early_stopping (10)
  - Larger batch_size (8)
  - Uses property_matched_v2 data (not old property_matched)
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')  # ensure before any CUDA init

import sys
import pandas as pd
import numpy as np
from unimol_tools.train import MolTrain

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "property_matched_v2", "data4finetune_0.csv")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_570m_v2")

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("Uni-Mol2 570M Fine-tuning v2 (Focal Loss)")
print(f"Data: {DATA_CSV}")
print(f"Save: {SAVE_DIR}")
print("=" * 60)

df = pd.read_csv(DATA_CSV)
print(f"Samples: {len(df)} (P={df['p_n'].sum()}, N={len(df)-df['p_n'].sum()})")

clf = MolTrain(
    task="multilabel_classification",
    data_type="molecule",
    model_name="unimolv2",
    model_size="570m",
    epochs=50,
    learning_rate=1e-4,
    batch_size=4,
    early_stopping=10,
    split="scaffold",
    kfold=5,
    save_path=SAVE_DIR,
    remove_hs=False,
    smiles_col="smiles",
    target_cols=["p_n"],
    use_cuda=True,
    use_amp=True,
    loss_key="focal",
)

print("\nStarting 570M fine-tuning with Focal Loss...")
result = clf.fit(DATA_CSV)
print(f"Result: {result}")
print(f"Done. Models saved to {SAVE_DIR}")
