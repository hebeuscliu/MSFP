#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整理 MSFP phase25 MD 全流程数据 + MM-GBSA 能量数据，输出统一汇总表 + 不合理检查。

数据链: 候选精选 -> MD(12体系100ns) -> 轨迹分析(RMSD/RMSF/H键) -> MM-GBSA(8体系 binding/dec/nmode)
只读文本/数据文件，不读图片。
"""
import os, json, csv
import numpy as np
import pandas as pd

D   = "/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking"
RES = f"{D}/results"
OUT = f"{D}/results"

STABLE  = [5614, 6115, 2732]          # ligRMSD<4
MARGIN  = [543, 6046, 6218, 6754, 5738] # 4-10
DISSOC  = [2532, 6117, 3188, 5743]     # >10 / 解离
BATCH1  = [2732, 6218, 6117, 2532, 5614, 6046]
BATCH2  = [543, 5743, 6115, 3188, 6754, 5738]
MMGBSA8 = [2732, 5614, 6115, 543, 6754, 5738, 6218, 6046]  # 算了能量的8体系

def tier(idx):
    if idx in STABLE: return "Stable"
    if idx in MARGIN: return "Marginal"
    return "Dissociated"

# ---- 1. 轨迹分析 ----
summ = pd.read_csv(f"{RES}/md_analysis_summary.csv")
summ["idx"] = summ["idx"].astype(int)
summ["batch"] = summ["idx"].apply(lambda i: "batch1" if i in BATCH1 else "batch2")
summ["tier"] = summ["idx"].apply(tier)

# ---- 2. MM-GBSA 能量 (8体系) ----
def read_delta(stat_file):
    """从 statistics 文件 DELTA 段读 GBTOT/VDW/GBELE 等 (精确 token 匹配, 避免GB误配GBELE/GBTOT)。"""
    if not os.path.isfile(stat_file): return {}
    out = {}
    f = 0
    keys = ("VDW","ELE","INT","GAS","GBSUR","GB","GBSOL","GBELE","GBTOT")
    with open(stat_file) as fh:
        for ln in fh:
            if "DELTA" in ln: f = 1; continue
            if f and ln.strip():
                toks = ln.split()
                if toks and toks[0] in keys:
                    out[toks[0]] = float(toks[1])
    return out

def read_tstot(e_dir):
    """熵项 DELTA TSTOT (= +TΔS, 负值因ΔS<0). 先3体系在 snapshot_statistics.out, 后5在 _nmode.out。
    返回 (TSTOT, src) — 注意 ΔG_bind = GBTOT - TSTOT。"""
    for fn in ("snapshot_statistics.out", "snapshot_statistics_nmode.out"):
        p = f"{e_dir}/{fn}"
        if not os.path.isfile(p): continue
        f = 0
        with open(p) as fh:
            for ln in fh:
                if "DELTA" in ln: f = 1; continue
                if f and ln.strip():
                    toks = ln.split()
                    if toks and toks[0] == "TSTOT":
                        return float(toks[1]), p
    return None, None

energy_rows = []
for idx in MMGBSA8:
    e = f"{D}/md/{idx}/energy"
    bind = read_delta(f"{e}/snapshot_statistics_binding.out")
    tstot, tsrc = read_tstot(e)
    dgtot = bind.get("GBTOT")
    dgbind = (dgtot - tstot) if (dgtot is not None and tstot is not None) else None
    energy_rows.append({
        "idx": idx, "tier": tier(idx),
        "dE_vdw": bind.get("VDW"), "dE_ele": bind.get("ELE"),
        "dG_gas": bind.get("GAS"), "dG_polar_GBELE": bind.get("GBELE"),
        "dG_TOT_noent": dgtot,
        "TSTOT": tstot, "nmode_src": os.path.basename(tsrc) if tsrc else None,
        "dG_bind_ent": dgbind,
    })
edf = pd.DataFrame(energy_rows)

# ---- 3. 合并 12 体系总表 ----
merged = summ.merge(edf.drop(columns=["tier"]), on="idx", how="left")
merged = merged.sort_values(["tier","idx"])
col_order = ["idx","batch","tier","prot_rmsd_mean","lig_rmsd_mean","lig_rmsd_last10ns_mean",
             "prot_rmsf_mean","hbond_avg_total",
             "dE_vdw","dE_ele","dG_gas","dG_polar_GBELE","dG_TOT_noent","TSTOT","dG_bind_ent"]
merged = merged[[c for c in col_order if c in merged.columns]]
merged.to_csv(f"{OUT}/md_energy_summary.csv", index=False)
merged.to_json(f"{OUT}/md_energy_summary.json", orient="records", indent=2)
print(f"== 汇总表 -> {OUT}/md_energy_summary.csv (.json) ==")
print(merged.to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "-"))

# ---- 4. 不合理检查 ----
print("\n" + "="*70 + "\n数据合理性检查\n" + "="*70)
issues = []
def issue(sev, msg): issues.append((sev, msg)); print(f"[{sev}] {msg}")

# (a) 候选列表一致性
cand = pd.read_csv(f"{RES}/md_candidates_final.csv")["idx"].astype(int).tolist()
md_all = sorted(BATCH1 + BATCH2)
missing_from_cand = [i for i in md_all if i not in cand]
extra_in_cand = [i for i in cand if i not in md_all]
if missing_from_cand:
    issue("WARN", f"MD跑了12体系但 md_candidates_final.csv 缺 {missing_from_cand} (候选表未更新含batch2新增)")
if extra_in_cand:
    issue("INFO", f"候选表有但未进MD: {extra_in_cand}")

# (b) 解离体系是否误算能量
for i in DISSOC:
    if i in MMGBSA8:
        issue("ERROR", f"解离体系 {i} 不应算 MM-GBSA (ligRMSD>10Å)")
issue("OK" if not any(i in MMGBSA8 for i in DISSOC) else "?", f"解离体系{DISSOC}均未算能量 (正确排除)")

# (c) ΔG 符号/范围
for _, r in edf.iterrows():
    if r["dG_TOT_noent"] is None: continue
    if r["dG_TOT_noent"] > 0: issue("WARN", f"{int(r['idx'])} ΔG_TOT={r['dG_TOT_noent']:.1f}>0 不合理(结合应为负)")
    if r["dG_bind_ent"] is not None and r["dG_bind_ent"] > 0:
        issue("WARN", f"{int(r['idx'])} ΔG_bind={r['dG_bind_ent']:.1f}>0 (含熵后不利结合)")
    if r["dG_bind_ent"] is not None and r["dG_bind_ent"] < -30:
        issue("WARN", f"{int(r['idx'])} ΔG_bind={r['dG_bind_ent']:.1f}<-30 过强, nmode熵可能异常")

# (d) 稳定性 vs ΔG 一致性
st = edf[edf["tier"]=="Stable"]["dG_bind_ent"].dropna()
mg = edf[edf["tier"]=="Marginal"]["dG_bind_ent"].dropna()
if len(st) and len(mg):
    print(f"\n[统计] Stable ΔG_bind 均值={st.mean():.2f} (n={len(st)}), Marginal 均值={mg.mean():.2f} (n={len(mg)})")
    if st.mean() < mg.mean():
        issue("OK", f"含熵ΔG: Stable({st.mean():.2f}) < Marginal({mg.mean():.2f}), 与动力学一致")
    else:
        issue("WARN", f"含熵ΔG: Stable({st.mean():.2f}) >= Marginal({mg.mean():.2f}), 与动力学不一致(nmode熵噪声)")
# 离群: 稳定但ΔG弱 / 漂移但ΔG强
for _, r in edf.iterrows():
    if r["dG_bind_ent"] is None: continue
    if r["tier"]=="Stable" and r["dG_bind_ent"] > -6:
        issue("WARN", f"稳定体系 {int(r['idx'])} ΔG_bind={r['dG_bind_ent']:.1f} 偏弱(>−6), nmode熵可疑")
    if r["tier"]=="Marginal" and r["dG_bind_ent"] < -11:
        issue("INFO", f"漂移体系 {int(r['idx'])} ΔG_bind={r['dG_bind_ent']:.1f} 偏强(<−11), 熵或构象采样所致")

# (e) nmode 熵范围 (TSTOT=+TΔS<0; −TΔS=-TSTOT>0 为熵惩罚)
ts = edf["TSTOT"].dropna()
neg_ts = -ts  # -TΔS, 正值=熵不利
print(f"\n[统计] −TΔS(熵惩罚) 范围: {neg_ts.min():.2f} ~ {neg_ts.max():.2f}, 均值={neg_ts.mean():.2f}")
if neg_ts.max() > 30:
    issue("WARN", f"最大熵惩罚 −TΔS={neg_ts.max():.1f}>30, nmode大体系熵偏大")
else:
    issue("INFO", f"熵惩罚 −TΔS 范围 {neg_ts.min():.1f}~{neg_ts.max():.1f} (nmode大体系常见+10~+30, 偏大但正常)")

# (f) 6754 轨迹拼接
p1 = f"{D}/md/6754/prod_p1.mdcrd"
if os.path.isfile(p1) and not os.path.isfile(f"{D}/md/6754/prod.mdcrd"):
    issue("WARN", "6754 有 prod_p1.mdcrd (中断重启段), 需确认分析用的是拼接后的100ns完整轨迹")

# (g) batch2 候选 docking/model 分数据缺失
cand_df = pd.read_csv(f"{RES}/md_candidates_final.csv")
miss_cand = [i for i in [543,5743,6115,3188,6754,5738] if i not in cand_df["idx"].astype(int).tolist()]
if miss_cand:
    issue("WARN", f"batch2体系 {miss_cand} 在 md_candidates_final.csv 无 docking/model/性质数据 (候选表未补全)")

# 写报告
with open(f"{OUT}/data_integrity_report.md","w") as f:
    f.write("# MSFP phase25 数据合理性检查报告\n\n")
    f.write(f"汇总表: `md_energy_summary.csv` / `.json`\n\n## 检查项\n\n")
    for sev, msg in issues:
        f.write(f"- **[{sev}]** {msg}\n")
    f.write(f"\n## 熵统计\n−TΔS(熵惩罚) 范围 {neg_ts.min():.2f}~{neg_ts.max():.2f}, 均值 {neg_ts.mean():.2f}\n")
    if len(st) and len(mg):
        f.write(f"\n## 稳定性 vs ΔG\nStable ΔG_bind 均值 {st.mean():.2f} (n={len(st)})\nMarginal ΔG_bind 均值 {mg.mean():.2f} (n={len(mg)})\n")
print(f"\n== 报告 -> {OUT}/data_integrity_report.md ==")
