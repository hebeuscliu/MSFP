#!/usr/bin/env python3
"""Phase25 配体准备: candidates_{ensemble,bestsingle,random}.csv -> 3D -> pdbqt (meeko)。

每条 SMILES: rdkit 加氢 + ETKDG3 3D embed + MMFF 优化 -> meeko mk_prepare -> pdbqt。
输出 ligands/{tag}/{idx}.pdbqt + manifest_{tag}.csv (status, pdbqt 路径, 原分数)。
"""
import os, sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
RDLogger.DisableLog('rdApp.*')

D = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(D, "data")
LIG = os.path.join(D, "ligands")
os.makedirs(LIG, exist_ok=True)

mk = MoleculePreparation()
writer = PDBQTWriterLegacy()

def make_pdbqt(smiles, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "bad_smiles"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
    if ok != 0:
        return None, "embed_fail"
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    try:
        preps = mk.prepare(mol)
    except Exception as e:
        return None, f"meeko_fail:{e}"
    if not preps:
        return None, "meeko_empty"
    try:
        pdbqt = writer.write_string(preps[0])[0]
    except Exception as e:
        return None, f"write_fail:{e}"
    return pdbqt, "ok"

for tag in ["ensemble", "bestsingle", "random"]:
    csv = os.path.join(DATA, f"candidates_{tag}.csv")
    df = pd.read_csv(csv)
    outdir = os.path.join(LIG, tag); os.makedirs(outdir, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        idx = int(r["idx"]); smi = r["smiles"]
        pdbqt, status = make_pdbqt(smi)
        path = ""
        if pdbqt is not None:
            path = os.path.join(outdir, f"{idx}.pdbqt")
            with open(path, "w") as f:
                f.write(pdbqt)
        rows.append(dict(idx=idx, name=r.get("name",""), smiles=smi, score=float(r["score"]),
                         mw=float(r["mw"]), status=status, pdbqt=path))
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(LIG, f"manifest_{tag}.csv"), index=False)
    n_ok = (man.status == "ok").sum()
    print(f"[{tag:10s}] {n_ok}/{len(man)} pdbqt ok | fail: {man[man.status!='ok']['status'].value_counts().to_dict()} | {time.time()-t0:.0f}s", flush=True)

print("Done.", flush=True)
