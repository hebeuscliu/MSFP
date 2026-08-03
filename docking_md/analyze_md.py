#!/usr/bin/env python3
"""Phase25 MD 轨迹分析: 对 6 体系 prod.mdcrd 跑 cpptraj (RMSD/RMSF/Hbond), 汇总。

每体系: 蛋白骨架 RMSD + 配体 RMSD (整体/末10ns 均值+最大) + 蛋白 RMSF + 配体-受体氢键。
输出 results/md_analysis_summary.csv + 各体系 cpptraj dat。
(不做 MM-GBSA, 跑完停下问用户。)

修正(v2):
  - hbond 关键字 angle(非 ang); 用 donormask/acceptormask 双向(P->L + L->P)
  - rmsd 加 nofit (否则默认重拟合 -> 恒0)
  - 配体用 :MOL (resname, 6体系均=残基301); 蛋白 :1-300&@CA
  - hbond avgout 解析: Frac 是第5列(f[4]), 非末列
"""
import os, subprocess, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
AMBER = "/root/disk1/senchenliu/home/amber24"
IDXS = [2732, 6218, 6117, 2532, 5614, 6046, 543, 5743, 6115, 3188, 6754, 5738]  # 全12体系 (batch1 6 + batch2 6)
PROT = ":1-300&@CA"
LIG = ":MOL"

def run_cpptraj(idx):
    d = os.path.join(D, "md", str(idx))
    prod = os.path.join(d, "prod.mdcrd")
    if not os.path.exists(prod):
        return None
    cppin = os.path.join(d, "analysis.cppin")
    with open(cppin, "w") as f:
        f.write(f"""parm {d}/complex_solvated.prmtop
trajin {prod}
autoimage
rms first {PROT}
rmsd {PROT} out {d}/rmsd_protein.dat nofit
rmsd {LIG} out {d}/rmsd_ligand.dat nofit
atomicfluct {PROT} out {d}/rmsf.dat byres
hbond donormask :1-300 acceptormask {LIG} out {d}/hbond_PL.dat avgout {d}/hbond_PL_avg.dat dist 3.0 angle 30.0
hbond donormask {LIG} acceptormask :1-300 out {d}/hbond_LP.dat avgout {d}/hbond_LP_avg.dat dist 3.0 angle 30.0
run
""")
    r = subprocess.run([f"{AMBER}/bin/cpptraj", "-i", cppin], capture_output=True, text=True,
                       env=dict(os.environ, AMBERHOME=AMBER), cwd=d)
    if r.returncode != 0 or "Error" in r.stderr:
        # 打印错误前几行便于诊断
        print(f"  [warn] idx {idx} cpptraj stderr: {r.stderr[:500]}", flush=True)
    return d

def parse_rmsd(dat):
    if not os.path.exists(dat): return None
    arr = np.loadtxt(dat, comments="#")
    if arr.ndim == 1: arr = arr[None, :]
    if arr.shape[1] < 2: return None
    return arr[:, 1]  # rmsd column

def parse_rmsf(dat):
    if not os.path.exists(dat): return None
    arr = np.loadtxt(dat, comments="#")
    if arr.ndim == 1: arr = arr[None, :]
    if arr.shape[1] < 2: return None
    return arr[:, 1]

def parse_hbond_avg(fn):
    """avgout 格式: #Acceptor DonorH Donor Frames Frac AvgDist AvgAng
    返回 [(label, frac, avgdist, avgang), ...] 按 Frac 降序"""
    if not os.path.exists(fn): return []
    rows = []
    for line in open(fn):
        if line.startswith("#") or not line.strip(): continue
        f = line.split()
        if len(f) >= 7:
            acc, don = f[0], f[2]
            frac, dist, ang = float(f[4]), float(f[5]), float(f[6])
            rows.append((f"{acc}-{don}", frac, dist, ang))
    return sorted(rows, key=lambda x: -x[1])[:5]

def parse_hbond_count(dat):
    """hbond timeseries: #Frame Nhbonds -> 均值"""
    if not os.path.exists(dat): return None
    arr = np.loadtxt(dat, comments="#")
    if arr.ndim == 1: arr = arr[None, :]
    if arr.shape[1] < 2: return None
    return float(np.mean(arr[:, 1]))

summary = []
for idx in IDXS:
    d = run_cpptraj(idx)
    if d is None:
        print(f"idx {idx}: prod.mdcrd 不存在, 跳过", flush=True); continue
    prot = parse_rmsd(os.path.join(d, "rmsd_protein.dat"))
    lig = parse_rmsd(os.path.join(d, "rmsd_ligand.dat"))
    rmsf = parse_rmsf(os.path.join(d, "rmsf.dat"))
    hb_PL = parse_hbond_avg(os.path.join(d, "hbond_PL_avg.dat"))
    hb_LP = parse_hbond_avg(os.path.join(d, "hbond_LP_avg.dat"))
    nPL = parse_hbond_count(os.path.join(d, "hbond_PL.dat"))
    nLP = parse_hbond_count(os.path.join(d, "hbond_LP.dat"))
    rec = {"idx": idx}
    if prot is not None:
        n10 = max(1, len(prot)//10)
        rec["prot_rmsd_mean"] = float(np.mean(prot)); rec["prot_rmsd_max"] = float(np.max(prot))
        rec["prot_rmsd_last10ns_mean"] = float(np.mean(prot[-n10:]))
    if lig is not None:
        n10 = max(1, len(lig)//10)
        rec["lig_rmsd_mean"] = float(np.mean(lig)); rec["lig_rmsd_max"] = float(np.max(lig))
        rec["lig_rmsd_last10ns_mean"] = float(np.mean(lig[-n10:]))
    if rmsf is not None:
        rec["prot_rmsf_mean"] = float(np.mean(rmsf)); rec["prot_rmsf_max"] = float(np.max(rmsf))
    rec["hbond_avg_total"] = (nPL or 0) + (nLP or 0)
    allhb = hb_PL + hb_LP
    rec["top_hbonds"] = " | ".join(f"{n}:{v:.2f}" for n, v, _, _ in sorted(allhb, key=lambda x: -x[1])[:3])
    summary.append(rec)
    print(f"idx {idx}: prot_RMSD(mean={rec.get('prot_rmsd_mean',0):.2f} last10ns={rec.get('prot_rmsd_last10ns_mean',0):.2f}) "
          f"lig_RMSD(mean={rec.get('lig_rmsd_mean',0):.2f} last10ns={rec.get('lig_rmsd_last10ns_mean',0):.2f}) "
          f"hb_avg={rec['hbond_avg_total']:.1f}", flush=True)

df = pd.DataFrame(summary)
out = os.path.join(D, "results", "md_analysis_summary.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}", flush=True)
print("\n=== MD 轨迹分析汇总 ===")
print(df.to_string(index=False))
print("\n[STOP] 轨迹分析完成, 未做 MM-GBSA, 等用户确认后再算结合自由能。", flush=True)
