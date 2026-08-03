#!/usr/bin/env python3
"""
Batch feature extraction for all 10 property-matched replicates.

Step 1: Extract MolCLR (GIN fine-tuned) features for each replicate
Step 2: Extract Uni-Mol2 features (after fine-tuning completes)

Usage:
  python scripts/extract_features_batch.py --modality molclr
  python scripts/extract_features_batch.py --modality unimol --ckpt-dir features/unimol/finetune_570m_v2
"""

import os, sys, argparse, csv
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data", "processed", "property_matched_v2")
OUT_DIR = DATA_DIR  # save features in same dir


def extract_molclr(args):
    """Extract MolCLR 512-dim features for all 10 replicates."""
    sys.path.insert(0, os.path.join(PROJ_ROOT, "features", "molclr"))
    from models.ginet_finetune import GINet
    from dataset.dataset_pred import MolTestDatasetWrapper

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load fine-tuned GIN
    model = GINet(task="classification", num_layer=5, emb_dim=300,
                  feat_dim=512, drop_ratio=0.3, pool="mean").to(device)

    # Load pretrained GIN weights
    pretrain_path = os.path.join(PROJ_ROOT, "features", "molclr", "ckpt",
                                 "pretrained_gin", "checkpoints", "model.pth")
    pretrain_state = torch.load(pretrain_path, map_location=device)
    model.load_my_state_dict(pretrain_state)

    # Load fine-tuned weights
    finetune_path = os.path.join(PROJ_ROOT, "features", "molclr", "finetune",
                                 "Jun05_01-04-55_BC_p_n", "checkpoints", "model.pth")
    finetune_state = torch.load(finetune_path, map_location=device)
    model.load_state_dict(finetune_state, strict=False)
    model.eval()
    print("Loaded fine-tuned GIN model.")

    for rep in range(10):
        csv_path = os.path.join(DATA_DIR, f"data4finetune_{rep}.csv")
        out_path = os.path.join(OUT_DIR, f"features_molclr_rep{rep}.npy")
        if os.path.exists(out_path) and not args.force:
            print(f"[{rep}] Already exists, skipping.")
            continue

        print(f"[{rep}] Extracting MolCLR features from {csv_path}...")
        dataset = MolTestDatasetWrapper(
            batch_size=32, num_workers=4, valid_size=0.0, test_size=0.0,
            data_path=csv_path, target="p_n", task="classification",
            splitting="random",
        )
        _, _, loader = dataset.get_data_loaders()

        embeddings = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                emb, _ = model(data)
                embeddings.append(emb.cpu().numpy())

        vecs = np.vstack(embeddings)
        np.save(out_path, vecs.astype(np.float32))
        print(f"[{rep}] Saved: {vecs.shape} → {out_path}")


def extract_unimol(args):
    """Extract Uni-Mol2 features for all 10 replicates + LMD."""
    from unimol_tools.predictor import UniMolRepr
    from rdkit import Chem

    device = f"cuda:{args.gpu.split(':')[-1]}" if "cuda" in args.gpu else args.gpu
    ckpt_dir = args.ckpt_dir

    print(f"Loading Uni-Mol2 model from {ckpt_dir}...")
    # Load fine-tuned model for feature extraction
    # UniMolRepr can load from a checkpoint directory
    try:
        model = UniMolRepr(model_name="unimolv2", model_size="570m",
                          load_model_dir=ckpt_dir, use_cuda=True)
    except Exception as e:
        print(f"Failed to load with load_model_dir: {e}")
        print("Trying pretrained 570M...")
        model = UniMolRepr(model_name="unimolv2", model_size="570m", use_cuda=True)

    # Extract for 10 replicates
    for rep in range(10):
        csv_path = os.path.join(DATA_DIR, f"data4finetune_{rep}.csv")
        out_path = os.path.join(OUT_DIR, f"features_unimol_rep{rep}.npy")
        if os.path.exists(out_path) and not args.force:
            print(f"[{rep}] Already exists, skipping.")
            continue

        print(f"[{rep}] Extracting Uni-Mol2 features...")
        df = pd.read_csv(csv_path)
        smiles = df["smiles"].tolist()

        # Filter invalid SMILES
        valid_smiles, valid_idx = [], []
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_idx.append(i)

        print(f"  Valid SMILES: {len(valid_smiles)}/{len(smiles)}")
        vecs = model.get_repr(valid_smiles, return_tensor=True)
        np.save(out_path, vecs.cpu().numpy().astype(np.float32))
        print(f"[{rep}] Saved: {vecs.shape} → {out_path}")

    # Extract for LMD (if requested)
    if args.lmd:
        lmd_csv = os.path.join(PROJ_ROOT, "data", "screening", "LMD_clean.csv")
        out_path = os.path.join(PROJ_ROOT, "data", "screening", "LMD_unimol.npy")
        if os.path.exists(out_path) and not args.force:
            print(f"LMD features already exist, skipping.")
            return

        print(f"Extracting Uni-Mol2 features for LMD library...")
        df = pd.read_csv(lmd_csv)
        smiles = df["smiles"].tolist()

        valid_smiles, valid_idx = [], []
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_idx.append(i)

        print(f"  Valid SMILES: {len(valid_smiles)}/{len(smiles)}")
        vecs = model.get_repr(valid_smiles, return_tensor=True)
        np.save(out_path, vecs.cpu().numpy().astype(np.float32))

        # Also save SMILES alignment
        pd.DataFrame({"smiles": valid_smiles}).to_csv(
            os.path.join(PROJ_ROOT, "data", "screening", "LMD_unimol_smiles.csv"),
            index=False,
        )
        print(f"Saved: {vecs.shape} → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["molclr", "unimol"])
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ckpt-dir", default=None,
                       help="Uni-Mol2 fine-tuned checkpoint dir")
    parser.add_argument("--lmd", action="store_true",
                       help="Also extract LMD library features (unimol only)")
    args = parser.parse_args()

    if args.modality == "molclr":
        extract_molclr(args)
    else:
        extract_unimol(args)
