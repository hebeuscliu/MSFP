#!/usr/bin/env python3
"""Phase25 候选准备 v2：先按训练分布把 LMD 预筛成类药子集，再取对接候选。

背景: phase24 直接在全 LMD 取 top, top100 全是 MW~1400 的超大糖苷天然产物 (OOD 尺寸偏差,
desc6 含 MW/HBD/HBA/TPSA 被外推), 100% 不类药、不可对接、与随机臂不可比。
训练集阳性中位 MW 432 (75%<=500), 故正确做法是先把 LMD 限到类药子集 (匹配训练分布),
再按 ensemble/bestsingle 分数取 top100; 随机臂也从类药子集抽, 保证两臂尺寸/类药性可比、可对接。

三组 (均类药):
  - ensemble  : 类药子集内 ensemble 分数 top100 (筛选臂 A1)
  - bestsingle: 类药子集内 bestsingle(rep2) 分数 top100 (筛选臂 A2)
  - random    : 类药子集内随机 100 (对照臂 B, seed=42)
"""
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog('rdApp.*')

P24 = "/root/disk1/senchenliu/MSFP/Final/phase24_mtssmol_fp_lmd_screen/results"
LMD_CLEAN = "/root/disk1/senchenliu/MSFP/Final/phase24_mtssmol_fp_lmd_screen/data/LMD_clean.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
N = 100

_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS = FilterCatalog(_params)

def props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol); logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol); hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol); tpsa = Descriptors.TPSA(mol)
    lip_viol = int(sum([mw > 500, logp > 5, hbd > 5, hba > 10]))
    veber = bool(rotb <= 10 and tpsa <= 140)
    pains = bool(PAINS.HasMatch(mol))
    druglike = bool(lip_viol <= 1 and veber and not pains and 150 <= mw <= 500)
    return dict(mw=round(mw,1), logp=round(logp,2), hbd=hbd, hba=hba, rotb=rotb,
                tpsa=round(tpsa,1), lip_viol=lip_viol, veber=veber, pains=pains, druglike=druglike)

# 全 LMD + 两套分数 (smiles 唯一, 用作 merge key; idx 在 LMD_clean 里不唯一, 不能用)
ens = pd.read_csv(os.path.join(P24, "LMD_ranked_ensemble.csv"))[["idx", "name", "smiles", "score"]].rename(columns={"score": "score_ens"})
bs = pd.read_csv(os.path.join(P24, "LMD_ranked_bestsingle.csv"))[["smiles", "score"]].rename(columns={"score": "score_bs"})
df = ens.merge(bs, on="smiles", validate="one_to_one")  # 26686 行, 每分子一行

print(f"LMD 全库: {len(df)} (unique smiles: {df.smiles.nunique()})", flush=True)
print("Computing ADMET/druglike on full LMD ...", flush=True)
plist = [props(s) for s in df["smiles"]]
valid = [p is not None for p in plist]
for k in ["mw","logp","hbd","hba","rotb","tpsa","lip_viol","veber","pains","druglike"]:
    df[k] = [p[k] if p else np.nan for p in plist]
df = df[valid].reset_index(drop=True)

n_dl = int(df["druglike"].sum())
print(f"类药子集 (MW150-500, Lip viol<=1, Veber, no PAINS): {n_dl} ({n_dl/len(df)*100:.0f}%)", flush=True)
dl = df[df["druglike"]].reset_index(drop=True)

# 三组候选 (类药子集内)
top_ens = dl.sort_values("score_ens", ascending=False).head(N).reset_index(drop=True)
top_bs = dl.sort_values("score_bs", ascending=False).head(N).reset_index(drop=True)
rng = np.random.RandomState(42)
rand = dl.sample(n=N, random_state=42).reset_index(drop=True)

def dump(tag, d, scol):
    out = d[["idx","name","smiles",scol,"mw","logp","hbd","hba","rotb","tpsa","lip_viol","veber","pains","druglike"]].copy()
    out = out.rename(columns={scol: "score"})
    out.to_csv(os.path.join(OUT, f"candidates_{tag}.csv"), index=False)
    print(f"[{tag:10s}] N={len(out)} MW med={out.mw.median():.0f} | score=[{out.score.min():.3f}..{out.score.max():.3f}] "
          f"| rotB med={out.rotb.median():.0f} | all Veber={out.veber.all()} | any PAINS={out.pains.any()}", flush=True)

print("\n=== 类药候选集 (可对接) ===", flush=True)
dump("ensemble", top_ens, "score_ens")
dump("bestsingle", top_bs, "score_bs")
dump("random", rand, "score_ens")  # random 用 ensemble 分数标注, 便于和 ensemble 臂比

# 三组重叠 (ensemble vs bestsingle top100)
ov = len(set(top_ens.idx) & set(top_bs.idx))
print(f"\nensemble∩bestsingle (类药 top100) = {ov}/100", flush=True)
print(f"Saved to {OUT}", flush=True)
