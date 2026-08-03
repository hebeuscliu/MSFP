#!/usr/bin/env python3
"""
Fine-tune Uni-Mol2 570M on SND1 property-matched data.
"""

import os, sys
import pandas as pd
import numpy as np
import torch
from unimol_tools.train import MolTrain

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(PROJECT_ROOT, "data/processed/property_matched/data4finetune_0.csv")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_570m")

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("Uni-Mol2 570M Fine-tuning on SND1")
print(f"Data: {DATA_CSV}")
print(f"Save: {SAVE_DIR}")
print("=" * 60)

df = pd.read_csv(DATA_CSV)
print(f"Samples: {len(df)} (P={df['p_n'].sum()}, N={len(df)-df['p_n'].sum()})")

clf = MolTrain(
    task="classification",
    data_type="molecule",
    model_name="unimolv2",
    model_size="570m",
    epochs=30,
    learning_rate=1e-4,
    batch_size=4,
    early_stopping=5,
    split="scaffold",
    kfold=5,
    save_path=SAVE_DIR,
    remove_hs=False,
    smiles_col="smiles",
    target_cols=["p_n"],
    use_cuda=True,
    use_amp=True,
)

print("\nStarting 570M fine-tuning...")
result = clf.fit(DATA_CSV)
print(f"Result: {result}")
print(f"Done. Models saved to {SAVE_DIR}")
