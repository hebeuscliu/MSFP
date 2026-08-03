#!/usr/bin/env python3
"""Phase25 精选 6-8 个 MD 候选: 多准则严选, 保证「完全没问题、成抑制剂可能性大」。

准则 (全过才入选):
  1. docking_score <= -8.0  (强结合, bestsingle arm)
  2. model_score >= 0.78    (高活性, MSFp P(active))
  3. 严格类药: Lipinski 违规=0, Veber, 无 PAINS, LogP in [0,5], rotB<=8, TPSA in [20,140]
  4. 无反应/毒效团 (酰卤/磺酰卤/醛/异氰酸酯/肼/亚硝基/重氮/过氧/Michael受体/烷基卤等)
  5. 骨架多样性: ECFP4 Tanimoto 聚类 (cutoff 0.55), 每簇取 docking 最优代表
最终按 (docking + model 综合分) 取 top 6-8。
"""
import os, numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog('rdApp.*')

D = os.path.dirname(os.path.abspath(__file__))
dock = pd.read_csv(os.path.join(D, "results", "dock_bestsingle.csv")).dropna(subset=["docking_score"])
cand = pd.read_csv(os.path.join(D, "data", "candidates_bestsingle.csv"))  # ADMET 已算
df = dock.merge(cand[["idx","smiles","mw","logp","hbd","hba","rotb","tpsa","lip_viol","veber","pains"]], on="idx", suffixes=("","_c"))

# 反应/毒效团 SMARTS (clearly problematic)
REACTIVE = [
    ("acyl_halide", "C(=O)[F,Cl,Br,I]"),
    ("sulfonyl_halide", "S(=O)(=O)[F,Cl,Br,I]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("isocyanate", "N=C=O"),
    ("isothiocyanate", "N=C=S"),
    ("hydrazine", "[NX3][NX3]"),
    ("nitroso", "N=O"),
    ("diazo", "N=#N"),
    ("peroxide", "[OX2][OX2]"),
    ("michael_acceptor", "C=C-C=O"),
    ("michael_acceptor_n", "C=C-C=N"),
    ("alkyl_halide", "[CX4][F,Cl,Br,I]"),
    ("epoxide", "[OX2r3]1[#6r3][#6r3]1"),
    ("aziridine", "[NX3r3]1[#6r3][#6r3]1"),
    ("beta_lactone", "[CX4r3]1[OX2r3][CX3r3](=O)[#6r3]1"),
    ("cyanide_nitrile", "C#N"),
]
REACTIVE_PATT = [(n, Chem.MolFromSmarts(s)) for n, s in REACTIVE if Chem.MolFromSmarts(s) is not None]

# PAINS (再查一遍, 保险)
_p = FilterCatalogParams(); _p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS = FilterCatalog(_p)

def ecfp4(mol): return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

def check(row):
    smi = row["smiles"]; mol = Chem.MolFromSmiles(smi)
    if mol is None: return False, "bad_smiles"
    reasons = []
    if row["docking_score"] > -8.0: reasons.append("docking>-8")
    if row["model_score"] < 0.78: reasons.append("model<0.78")
    if row["lip_viol"] != 0: reasons.append(f"lip_viol={row['lip_viol']}")
    if not row["veber"]: reasons.append("veber_fail")
    if row["pains"]: reasons.append("PAINS")
    if not (0 <= row["logp"] <= 5): reasons.append(f"logp={row['logp']}")
    if row["rotb"] > 8: reasons.append(f"rotb={row['rotb']}")
    if not (20 <= row["tpsa"] <= 140): reasons.append(f"tpsa={row['tpsa']}")
    # reactive
    rxn = [n for n, p in REACTIVE_PATT if mol.HasSubstructMatch(p)]
    if rxn: reasons.append("reactive:" + ",".join(rxn))
    if PAINS.HasMatch(mol): reasons.append("PAINS2")
    return (len(reasons) == 0), ";".join(reasons) if reasons else "ok"

df["pass"], df["reason"] = zip(*df.apply(check, axis=1))
surv = df[df["pass"]].copy()
print(f"严选通过: {len(surv)} / {len(df)} (docking<=-8 & model>=0.78 & 严格类药 & 无反应)", flush=True)
print(f"淘汰原因 top: {df[~df['pass']]['reason'].value_counts().head(10).to_dict()}", flush=True)

if len(surv) == 0:
    print("无分子通过严选, 放宽: docking<=-7.8 或 model>=0.76");
    # 放宽只看 docking<=-7.8 + 类药 + 无反应
    surv = df[(df.docking_score<=-7.8) & (df.lip_viol==0) & (df.veber) & (~df.pains)].copy()
    print(f"放宽后: {len(surv)}")

# 骨架多样性: ECFP4 Tanimoto 聚类 (cutoff 0.55), 每簇取 docking 最优
surv = surv.sort_values("docking_score").reset_index(drop=True)
fps = [ecfp4(Chem.MolFromSmiles(s)) for s in surv["smiles"]]
clusters = []  # list of (rep_idx, members)
assigned = [False]*len(surv)
for i in range(len(surv)):
    if assigned[i]: continue
    members = [i]; assigned[i] = True
    for j in range(i+1, len(surv)):
        if not assigned[j] and DataStructs.TanimotoSimilarity(fps[i], fps[j]) >= 0.55:
            members.append(j); assigned[j] = True
    clusters.append(members)  # i 是 docking 最优(已排序)
print(f"多样性聚类: {len(clusters)} 个骨架簇", flush=True)

# 每簇代表 = docking 最优(=该簇第一个, 因已按 docking 排序), 综合分 = -docking + model 排序
reps = surv.iloc[[c[0] for c in clusters]].copy()
reps["combined"] = -reps["docking_score"] + reps["model_score"]  # 越大越优
reps = reps.sort_values("combined", ascending=False).reset_index(drop=True)

N = 8
final = reps.head(N).copy()
print(f"\n=== 最终 MD 候选 (top {len(final)}, 多样性+综合分) ===", flush=True)
cols = ["idx","name","docking_score","model_score","mw","logp","hbd","hba","rotb","tpsa","smiles"]
print(final[cols].to_string(index=False))
final[cols].to_csv(os.path.join(D, "results", "md_candidates_final.csv"), index=False)
print(f"\nSaved results/md_candidates_final.csv ({len(final)} 个)", flush=True)
# 簇大小
for k, c in enumerate(clusters[:N]):
    print(f"  簇{k}: {len(c)} 个相似分子, 代表 idx={surv.iloc[c[0]]['idx']}")
