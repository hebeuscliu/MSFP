#!/usr/bin/env python3
"""Phase25 Vina 对接: 对一个 tag 的候选批量对接 SND1 主口袋。

maps 只算一次复用; 每配体 dock(exhaustiveness=8, n_poses=10), 记 best affinity。
增量写 results/dock_{tag}.csv (idx,name,smiles,model_score,mw,docking_score,status)。
用法: python dock.py <tag>   tag in {ensemble,bestsingle,random}
"""
import os, sys, time
import pandas as pd
from vina import Vina

tag = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(D, "receptor", "snd1.pdbqt")
LIG = os.path.join(D, "ligands")
RES = os.path.join(D, "results"); os.makedirs(RES, exist_ok=True)
BOX_CENTER = [47.42, 58.55, -53.78]
BOX_SIZE = [30.0, 30.0, 25.8]

man = pd.read_csv(os.path.join(LIG, f"manifest_{tag}.csv"))
out_csv = os.path.join(RES, f"dock_{tag}.csv")

v = Vina(sf_name='vina', cpu=8, verbosity=0)
v.set_receptor(REC)
v.compute_vina_maps(center=BOX_CENTER, box_size=BOX_SIZE)
print(f"[{tag}] maps ready, docking {len(man)} ligands (cpu=8, ex=8)...", flush=True)

rows = []
t0 = time.time()
for i, r in man.iterrows():
    idx = int(r['idx']); pdbqt = r['pdbqt']
    score = None; status = "ok"
    if not isinstance(pdbqt, str) or not pdbqt or not os.path.exists(pdbqt):
        status = "no_pdbqt"
    else:
        try:
            v.set_ligand_from_file(pdbqt)
            v.dock(exhaustiveness=8, n_poses=10)
            e = v.energies()
            score = float(e[0][0]) if e is not None and len(e) > 0 else None
            if score is None:
                status = "no_energy"
        except Exception as ex:
            status = f"err:{type(ex).__name__}:{ex}"
    rows.append(dict(idx=idx, name=r.get('name', ''), smiles=r['smiles'],
                     model_score=float(r['score']), mw=float(r['mw']),
                     docking_score=score, status=status))
    if (i + 1) % 10 == 0 or i == len(man) - 1:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        done = sum(1 for x in rows if x['docking_score'] is not None)
        print(f"[{tag}] {i+1}/{len(man)} ({done} scored) {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(rows).to_csv(out_csv, index=False)
n_ok = sum(1 for x in rows if x['docking_score'] is not None)
print(f"[{tag}] DONE {n_ok}/{len(rows)} scored -> {out_csv}", flush=True)
