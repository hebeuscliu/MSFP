#!/usr/bin/env python3
"""分析 9 个 SND1-配体复合物的结合位点，定义 vina grid box。

8 个文件夹共用受体 00a0b472 (9-new,14-new,38-new,48-new,58-new,84,119-new,9) -> 同坐标架,
配体中心直接可比。67/86/163 用不同受体, 用 gemmi CA 超叠对齐到参考架再比。
主口袋 = 多数配体聚类位点 -> grid box 中心+尺寸 + 结合残基。
"""
import os, glob, numpy as np, gemmi

FILES = "/root/disk1/senchenliu/MSFP-MD-files"
REF_RECEPTOR = os.path.join(FILES, "9-new", "receptor.pdb")  # 00a0b472, 参考架
# 同架 (受体==00a0b472) 的复合物
SAME_FRAME = ["9-new", "14-new", "38-new", "48-new", "58-new", "84", "119-new", "9"]
# 异架 (需对齐)
ALIGN = ["67-new", "86-new", "163-new"]

def load_atoms(path):
    st = gemmi.read_structure(path)
    lig, rec = [], []
    for model in st:
        for chain in model:
            for res in chain:
                is_lig = (res.name == "UNL")
                for atom in res:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    if is_lig:
                        lig.append(p)
                    else:
                        rec.append((res.seqid.num, res.name, atom.name, p))
    return np.array(lig), rec

def ca_coords(rec):
    return np.array([p for (n,rn,an,p) in rec if an == "CA"])

def receptor_residues(rec):
    return rec  # list of (seqid, resname, atomname, pos)

print(f"参考受体: {REF_RECEPTOR}", flush=True)
ref_lig, ref_rec = load_atoms(os.path.join(FILES, "9-new", "complex.pdb"))
ref_ca = ca_coords(ref_rec)

centers = {}      # tag -> ligand center (ref frame)
all_lig_atoms = {} # tag -> all ligand atom coords (ref frame)
for tag in SAME_FRAME:
    cpath = os.path.join(FILES, tag, "complex.pdb")
    if not os.path.exists(cpath):
        cpath = os.path.join(FILES, tag, "complex_check.pdb")
    lig, rec = load_atoms(cpath)
    if len(lig) == 0:
        print(f"  [warn] {tag}: no UNL ligand in {cpath}"); continue
    centers[tag] = lig.mean(axis=0)
    all_lig_atoms[tag] = lig
    print(f"  {tag:8s}: ligand atoms={len(lig)} center=({lig[:,0].mean():.1f},{lig[:,1].mean():.1f},{lig[:,2].mean():.1f})", flush=True)

# 对齐异架复合物
print("\n对齐异架复合物 (67/86/163) ...", flush=True)
for tag in ALIGN:
    cpath = os.path.join(FILES, tag, "complex.pdb")
    if not os.path.exists(cpath):
        continue
    lig, rec = load_atoms(cpath)
    mob_ca = ca_coords(rec)
    if len(mob_ca) == 0 or len(lig) == 0:
        print(f"  [warn] {tag}: no CA or ligand"); continue
    n = min(len(ref_ca), len(mob_ca))
    # gemmi superposition (CA, 同蛋白)
    rp = [gemmi.Position(float(x),float(y),float(z)) for x,y,z in ref_ca[:n]]
    mp = [gemmi.Position(float(x),float(y),float(z)) for x,y,z in mob_ca[:n]]
    try:
        sup = gemmi.calculate_superposition(rp, mp, gemmi.PolType.Ca)
        tr = sup.transform
        lig_t = np.array([[tr.mat[0][0]*x+tr.mat[0][1]*y+tr.mat[0][2]*z+tr.vec.x,
                           tr.mat[1][0]*x+tr.mat[1][1]*y+tr.mat[1][2]*z+tr.vec.y,
                           tr.mat[2][0]*x+tr.mat[2][1]*y+tr.mat[2][2]*z+tr.vec.z] for x,y,z in lig])
        centers[tag] = lig_t.mean(axis=0)
        all_lig_atoms[tag] = lig_t
        print(f"  {tag:8s}: aligned, ligand atoms={len(lig)} center=({lig_t[:,0].mean():.1f},{lig_t[:,1].mean():.1f},{lig_t[:,2].mean():.1f}) rmsd={sup.rmsd:.2f}", flush=True)
    except Exception as e:
        print(f"  [warn] {tag}: align failed {e}", flush=True)

# 聚类: 以 9-new 为种子, 距离<12A 视为同位点
tags = list(centers.keys())
C = np.array([centers[t] for t in tags])
print("\n=== 配体中心两两距离 (Å) ===", flush=True)
print("        " + " ".join(f"{t[:6]:>7s}" for t in tags))
for i, ti in enumerate(tags):
    print(f"{ti[:6]:>7s} " + " ".join(f"{np.linalg.norm(C[i]-C[j]):7.1f}" for j in range(len(tags))))

# 主簇: 从 9-new 出发, 距离<12A
seed = "9-new" if "9-new" in centers else tags[0]
si = tags.index(seed)
main = [t for t in tags if np.linalg.norm(centers[seed] - centers[t]) < 12.0]
outliers = [t for t in tags if t not in main]
print(f"\n主口袋簇 (<12A from {seed}): {main}", flush=True)
print(f"其他位点: {outliers}", flush=True)

# grid box: 主簇所有配体原子 extent + 7A padding, 单轴上限 30
M = np.vstack([all_lig_atoms[t] for t in main if t in all_lig_atoms])
lo, hi = M.min(axis=0), M.max(axis=0)
pad = 7.0
size = np.minimum((hi - lo) + 2*pad, 30.0)
center = (lo + hi) / 2.0
print(f"\n=== GRID BOX (主口袋, 参考 9-new/receptor.pdb 架) ===", flush=True)
print(f"center_x = {center[0]:.2f}", flush=True)
print(f"center_y = {center[1]:.2f}", flush=True)
print(f"center_z = {center[2]:.2f}", flush=True)
print(f"size_x = {size[0]:.1f}  size_y = {size[1]:.1f}  size_z = {size[2]:.1f}  (Å, vina)", flush=True)

# 结合残基: 参考架受体中距主簇配体原子 <5A 的残基
rec_all = ref_rec  # 9-new complex 的受体 (==00a0b472)
main_atoms = M
contact = {}
for (seqid, rn, an, p) in rec_all:
    d = np.linalg.norm(main_atoms - p, axis=1).min()
    if d < 5.0:
        contact[(seqid, rn)] = min(contact.get((seqid,rn),9), d)
print(f"\n主口袋结合残基 (距配体<5Å, {len(contact)} 个):", flush=True)
print(" ".join(f"{rn}{seqid}" for (seqid,rn),d in sorted(contact.items())), flush=True)

# 保存 grid box 配置
import json
cfg = {"receptor_ref": REF_RECEPTOR, "main_cluster": main, "outliers": outliers,
       "center": [float(center[0]),float(center[1]),float(center[2])],
       "size": [float(size[0]),float(size[1]),float(size[2])],
       "binding_residues": [{"seqid":s,"resname":r,"mindist":float(d)} for (s,r),d in sorted(contact.items())]}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "grid_box.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
print(f"\nSaved {out}", flush=True)
