#!/usr/bin/env python3
"""Phase25 MD 体系准备 (6 个候选): pose->antechamber->tleap, 每个建溶剂化复合物。

对每个 idx: 重对接(ex=32)存 pose -> mk_export sdf -> antechamber(gaff+AM1-BCC) ->
parmchk2 frcmod -> tleap 建复合物。输出 md/{idx}/complex_solvated.prmtop/inpcrd。
"""
import os, sys, subprocess, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from vina import Vina
RDLogger.DisableLog('rdApp.*')

D = os.path.dirname(os.path.abspath(__file__))
REC_PDBQT = os.path.join(D, "receptor", "snd1.pdbqt")
LIG_DIR = os.path.join(D, "ligands", "bestsingle")
MD = os.path.join(D, "md")
REF_RECEPTOR = "/root/disk1/senchenliu/MSFP-MD-files/9-new/receptor.pdb"
AMBER = "/root/disk1/senchenliu/home/amber24"
MKEEKO = "/root/miniconda3/envs/pytorch/bin/mk_export.py"
PY = "/root/miniconda3/envs/pytorch/bin/python"
BOX_CENTER = [47.42, 58.55, -53.78]; BOX_SIZE = [30.0, 30.0, 25.8]
IDXS = [543, 5743, 6115, 3188, 6754, 5738]  # batch2: 6 新候选 (原6已完成会自动跳过)

def charge_from_sdf(sdf):
    m = Chem.MolFromMolFile(sdf)
    if m is None: return None
    try: Chem.SanitizeMol(m)
    except: pass
    return Chem.GetFormalCharge(m)

def charge_from_pdbqt(pdbqt):
    s = 0.0
    for line in open(pdbqt):
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            try: s += float(f[-2])  # meeko: 末第二字段=partial charge
            except: pass
    return int(round(s))

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

for idx in IDXS:
    d = os.path.join(MD, str(idx)); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    print(f"\n=== idx {idx} ===", flush=True)
    # 跳过已完成的 (有 complex_solvated.prmtop)
    if os.path.exists(os.path.join(d, "complex_solvated.prmtop")):
        print(f"  已有 complex_solvated.prmtop, 跳过", flush=True); continue

    # 1. 重对接存 pose
    inpdbqt = os.path.join(LIG_DIR, f"{idx}.pdbqt")
    pose = os.path.join(d, f"{idx}_pose.pdbqt")
    if not os.path.exists(pose):
        v = Vina(sf_name='vina', cpu=8, verbosity=0)
        v.set_receptor(REC_PDBQT)
        v.compute_vina_maps(center=BOX_CENTER, box_size=BOX_SIZE)
        v.set_ligand_from_file(inpdbqt)
        v.dock(exhaustiveness=32, n_poses=10)
        v.write_poses(pose, n_poses=10, overwrite=True)
        print(f"  pose: {float(v.energies()[0][0]):.3f}", flush=True)

    # 2. mk_export -> sdf
    sdf = os.path.join(d, f"{idx}_pose.sdf")
    if not os.path.exists(sdf):
        r = run([PY, MKEEKO, pose, "-s", sdf])
        if r.returncode != 0: print(f"  mk_export fail: {r.stderr[:200]}", flush=True)

    # 3. 净电荷
    nc = charge_from_sdf(sdf)
    if nc is None: nc = charge_from_pdbqt(inpdbqt)
    print(f"  net charge: {nc}", flush=True)

    # 4. antechamber (gaff + AM1-BCC)
    env = dict(os.environ, AMBERHOME=AMBER)
    if not os.path.exists(os.path.join(d, "ligand.mol2")):
        r = run([f"{AMBER}/bin/antechamber", "-i", sdf, "-fi", "sdf", "-o", "ligand.mol2",
                 "-fo", "mol2", "-c", "bcc", "-nc", str(nc), "-m", "1", "-s", "2", "-at", "gaff"],
                cwd=d, env=env, timeout=900)
        if r.returncode != 0: print(f"  antechamber fail: {r.stderr[:300]}", flush=True)
    # 5. parmchk2
    if not os.path.exists(os.path.join(d, "ligand.frcmod")):
        r = run([f"{AMBER}/bin/parmchk2", "-i", "ligand.mol2", "-f", "mol2", "-o", "ligand.frcmod", "-s", "gaff"],
                cwd=d, env=env)
        if r.returncode != 0: print(f"  parmchk2 fail: {r.stderr[:200]}", flush=True)

    # 6. receptor.pdb
    rec = os.path.join(d, "receptor.pdb")
    if not os.path.exists(rec):
        subprocess.run(["cp", REF_RECEPTOR, rec])

    # 7. tleap
    tleap_in = os.path.join(d, "tleap.in")
    with open(tleap_in, "w") as f:
        f.write("""source leaprc.protein.ff19SB
source leaprc.water.tip3p
source leaprc.gaff
loadamberparams ligand.frcmod
UNL = loadmol2 ligand.mol2
check UNL
saveoff UNL ligand.lib
loadoff ligand.lib
rec = loadpdb receptor.pdb
lig = loadmol2 ligand.mol2
com = combine {rec lig}
set default PBRadii mbondi2
saveamberparm com complex.prmtop complex.inpcrd
savepdb com complex_check.pdb
addions com Na+ 0
addions com Cl- 0
solvatebox com TIP3PBOX 10.0
savepdb com complex_solvated_check.pdb
saveamberparm com complex_solvated.prmtop complex_solvated.inpcrd
quit
""")
    if not os.path.exists(os.path.join(d, "complex_solvated.prmtop")):
        r = run([f"{AMBER}/bin/tleap", "-f", "tleap.in"], cwd=d, env=env, timeout=300)
        ok = os.path.exists(os.path.join(d, "complex_solvated.prmtop"))
        print(f"  tleap: {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)
        if not ok: print(f"    {r.stdout[-300:] if r.stdout else ''}", flush=True)
    else:
        print(f"  tleap 已完成 ({time.time()-t0:.0f}s)", flush=True)

print("\n=== 汇总 ===", flush=True)
for idx in IDXS:
    d = os.path.join(MD, str(idx))
    ok = os.path.exists(os.path.join(d, "complex_solvated.prmtop"))
    print(f"  idx {idx}: {'OK' if ok else 'MISSING'}", flush=True)
