#!/usr/bin/env python3
"""Phase25 对接富集分析: 筛选臂 (ensemble/bestsingle) vs 随机臂 (random) 的 SND1 docking 打分。

可行性判据: 筛选臂 docking 打分是否系统性低于(优于)随机臂 (Mann-Whitney, 单侧 less)。
更负 = 更好结合。EF = 筛选臂 hit 率 / 随机臂 hit 率 (hit = score<=阈值)。
"""
import os, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(D, "results")
try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def load(tag):
    df = pd.read_csv(os.path.join(RES, f"dock_{tag}.csv"))
    df = df.dropna(subset=["docking_score"])
    df["docking_score"] = df["docking_score"].astype(float)
    return df

ens, bs, rand = load("ensemble"), load("bestsingle"), load("random")
print(f"n: ensemble={len(ens)} bestsingle={len(bs)} random={len(rand)}", flush=True)

print("\n=== docking 打分分布 (kcal/mol, 更负=更好) ===", flush=True)
print(f"{'arm':12s} {'median':>8s} {'mean':>8s} {'min':>7s} {'max':>7s} {'<=-7':>5s} {'<=-8':>5s}")
for name, d in [("ensemble", ens), ("bestsingle", bs), ("random", rand)]:
    s = d["docking_score"]
    print(f"{name:12s} {s.median():8.2f} {s.mean():8.2f} {s.min():7.2f} {s.max():7.2f} {(s<=-7).sum():5d} {(s<=-8).sum():5d}")

print("\n=== Mann-Whitney U (筛选臂 < 随机臂, 单侧 less=更好) ===", flush=True)
for name, d in [("ensemble", ens), ("bestsingle", bs)]:
    if HAVE_SCIPY:
        u, p = mannwhitneyu(d["docking_score"], rand["docking_score"], alternative="less")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{name:12s} vs random: median {d.docking_score.median():.2f} vs {rand.docking_score.median():.2f}  p={p:.4g} {sig}")
    else:
        # 简单 manual rank-sum 近似
        from scipy import stats  # noqa
        print(f"{name:12s}: scipy 不可用, 仅看 median {d.docking_score.median():.2f} vs {rand.docking_score.median():.2f}")

print("\n=== 富集因子 EF (hit = score<=阈值) ===", flush=True)
print(f"{'thr':>6s} {'ens%':>6s} {'bs%':>6s} {'rand%':>6s} {'EF_ens':>7s} {'EF_bs':>7s}")
for thr in [-7.0, -7.5, -8.0, -8.5]:
    er = (rand["docking_score"] <= thr).mean()
    ee = (ens["docking_score"] <= thr).mean()
    eb = (bs["docking_score"] <= thr).mean()
    print(f"{thr:6.1f} {ee*100:6.0f} {eb*100:6.0f} {er*100:6.0f} {ee/er if er>0 else float('nan'):7.2f} {eb/er if er>0 else float('nan'):7.2f}")

print("\n=== ensemble vs bestsingle (哪套筛选策略 docking 更好) ===", flush=True)
if HAVE_SCIPY:
    u, p = mannwhitneyu(ens["docking_score"], bs["docking_score"], alternative="less")
    print(f"ensemble median {ens.docking_score.median():.2f} vs bestsingle {bs.docking_score.median():.2f}  p={p:.4g} (less: ens 更优)")
ov = len(set(ens.nsmallest(20,'docking_score')['idx']) & set(bs.nsmallest(20,'docking_score')['idx']))
print(f"top-20 docking hit 重叠 ensemble∩bestsingle = {ov}/20")

print("\n=== top 10 docking hits (ensemble) ===", flush=True)
print(ens.nsmallest(10, "docking_score")[["idx","name","docking_score","model_score","mw"]].to_string(index=False))
print("\n=== top 10 docking hits (bestsingle) ===", flush=True)
print(bs.nsmallest(10, "docking_score")[["idx","name","docking_score","model_score","mw"]].to_string(index=False))

# 保存汇总
summ = {
    "ensemble_median": float(ens.docking_score.median()), "bestsingle_median": float(bs.docking_score.median()),
    "random_median": float(rand.docking_score.median()),
    "ensemble_mean": float(ens.docking_score.mean()), "bestsingle_mean": float(bs.docking_score.mean()),
    "random_mean": float(rand.docking_score.mean()),
}
import json
with open(os.path.join(RES, "docking_summary.json"), "w") as f:
    json.dump(summ, f, indent=2)
print(f"\nSaved {os.path.join(RES,'docking_summary.json')}", flush=True)
