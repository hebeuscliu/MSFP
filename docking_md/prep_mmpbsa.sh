#!/bin/bash
# Prepare MM-GBSA for 3 stable systems (5614, 6115, 2732):
#   - receptor.prmtop / ligand.prmtop (split from complex.prmtop via cpptraj parmstrip)
#   - prod_strip.mdcrd (final 20ns, every 10th frame = 200 snapshots, autoimage+strip water)
#   - 4 .mmpbsa files per system (from 9-new templates, per-system ligand range)
set -e
AMBER=/root/disk1/senchenliu/home/amber24
D=/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking
TPL=/root/disk1/senchenliu/MSFP-MD-files/9-new/energy

# idx: LSTART LSTOP  (receptor 1-4811, ligand 4812-LSTOP; NTOTAL=LSTOP after strip)
LIG_start=(5614 6115 2732)
declare -A LSTOP=( [5614]=4875 [6115]=4871 [2732]=4873 )

for idx in 5614 6115 2732; do
  lstop=${LSTOP[$idx]}; lstart=4812; ntotal=$lstop
  w=$D/md/$idx/energy; mkdir -p $w
  cd $D/md/$idx
  echo "=== idx $idx (LSTART=$lstart LSTOP=$lstop NTOTAL=$ntotal) ==="

  # 1. stripped traj: final 20ns (frames 8001-10000), every 10th = 200 frames, autoimage, strip water/ions
  if [ ! -f $w/prod_strip.mdcrd ]; then
    cat > /tmp/strip_$idx.in <<EOF
parm complex_solvated.prmtop
trajin prod.mdcrd 8001 10000 10
autoimage
strip :WAT,Cl-,Na+
trajout $w/prod_strip.mdcrd netcdf
run
EOF
    $AMBER/bin/cpptraj -i /tmp/strip_$idx.in 2>&1 | grep -iE "processed [0-9]+|error" | tail -2
  fi

  # 2. receptor.prmtop (strip MOL from unsolvated complex)
  if [ ! -f receptor.prmtop ]; then
    cat > /tmp/prec_$idx.in <<EOF
parm complex.prmtop
parmstrip :MOL
parmwrite out receptor.prmtop
EOF
    $AMBER/bin/cpptraj -i /tmp/prec_$idx.in 2>&1 | grep -iE "wrote|error" | tail -1
  fi

  # 3. ligand.prmtop (strip receptor 1-300 from complex, leaving MOL=301)
  if [ ! -f ligand.prmtop ]; then
    cat > /tmp/plig_$idx.in <<EOF
parm complex.prmtop
parmstrip :1-300
parmwrite out ligand.prmtop
EOF
    $AMBER/bin/cpptraj -i /tmp/plig_$idx.in 2>&1 | grep -iE "wrote|error" | tail -1
  fi

  # 4. extract_coords.mmpbsa (GC=1): per-system NTOTAL/LSTART/LSTOP, NSTART=1 NSTOP=200
  sed -e "s|^NTOTAL .*|NTOTAL                $ntotal|" \
      -e "s|^NSTART .*|NSTART                1|" \
      -e "s|^NSTOP .*|NSTOP                 200|" \
      -e "s|^LSTART .*|LSTART                $lstart|" \
      -e "s|^LSTOP .*|LSTOP                 $lstop|" \
      -e "s|^RSTART .*|RSTART                1|" \
      -e "s|^RSTOP .*|RSTOP                 4811|" \
      -e "s|^TRAJECTORY .*|TRAJECTORY            $w/prod_strip.mdcrd|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" \
      -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" \
      -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/extract_coords.mmpbsa > $w/extract_coords.mmpbsa

  # 5. binding_energy.mmpbsa (MM=1 GB=1 IGB=2): START=1 STOP=200, PARALLEL=8
  sed -e "s|^PARALLEL .*|PARALLEL              8|" \
      -e "s|^START .*|START                 1|" \
      -e "s|^STOP .*|STOP                  200|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" \
      -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" \
      -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/binding_energy.mmpbsa > $w/binding_energy.mmpbsa

  # 6. dec_res.mmpbsa (DC=1 DCTYPE=2 IGB=2 GBSA=2): START=1 STOP=200, PARALLEL=8
  sed -e "s|^START .*|START                 1|" \
      -e "s|^STOP .*|STOP                  200|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" \
      -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" \
      -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/dec_res.mmpbsa > $w/dec_res.mmpbsa

  # 7. nmode.mmpbsa (NM=1 vacuum DIELC=4): last 10 frames (START=191 STOP=200)
  sed -e "s|^START .*|START                 191|" \
      -e "s|^STOP .*|STOP                  200|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" \
      -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" \
      -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/nmode.mmpbsa > $w/nmode.mmpbsa

  nf=$(ls $w/snapshot_com.crd.* 2>/dev/null | wc -l)
  echo "  -> $w: $(ls $w/*.mmpbsa | wc -l) mmpbsa | prod_strip.mdcrd $(du -h $w/prod_strip.mdcrd 2>/dev/null | cut -f1) | receptor/ligand.prmtop $([ -f receptor.prmtop ] && echo ✅) $([ -f ligand.prmtop ] && echo ✅)"
done
echo "=== prep 完成 ==="