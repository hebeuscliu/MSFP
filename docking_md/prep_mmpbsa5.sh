#!/bin/bash
# Prep MM-GBSA for 5 marginal systems (543,6046,6218,6754,5738):
#   stable window per system (not final 20ns, since ligand drifts late)
set -e
AMBER=/root/disk1/senchenliu/home/amber24
D=/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking
TPL=/root/disk1/senchenliu/MSFP-MD-files/9-new/energy

declare -A LSTOP=( [543]=4874 [6046]=4864 [6218]=4884 [6754]=4868 [5738]=4870 )
declare -A WSTART=( [543]=1 [6046]=1 [6218]=1 [6754]=3001 [5738]=1001 )
declare -A WEND=( [543]=2000 [6046]=2000 [6218]=2000 [6754]=5000 [5738]=3000 )

for idx in 543 6046 6218 6754 5738; do
  lstop=${LSTOP[$idx]}; ws=${WSTART[$idx]}; we=${WEND[$idx]}; ntotal=$lstop
  w=$D/md/$idx/energy; mkdir -p $w/snapshot
  cd $D/md/$idx
  echo "=== idx $idx (LSTOP=$lstop 窗口=$ws-$we = 2000帧) ==="
  # stripped traj (stable window, ASCII)
  cat > /tmp/strip_$idx.in <<EOF
parm complex_solvated.prmtop
trajin prod.mdcrd $ws $we 1
autoimage
strip :WAT,Cl-,Na+
trajout $w/prod_strip.mdcrd
run
EOF
  $AMBER/bin/cpptraj -i /tmp/strip_$idx.in 2>&1 | grep -iE "processed [0-9]+|error" | tail -1
  # receptor/ligand prmtops
  [ -f receptor.prmtop ] || cat > /tmp/prec_$idx.in <<EOF
parm complex.prmtop
parmstrip :MOL
parmwrite out receptor.prmtop
EOF
  [ -f receptor.prmtop ] || $AMBER/bin/cpptraj -i /tmp/prec_$idx.in 2>&1 | tail -1
  [ -f ligand.prmtop ] || cat > /tmp/plig_$idx.in <<EOF
parm complex.prmtop
parmstrip :1-300
parmwrite out ligand.prmtop
EOF
  [ -f ligand.prmtop ] || $AMBER/bin/cpptraj -i /tmp/plig_$idx.in 2>&1 | tail -1
  # 4 .mmpbsa
  sed -e "s|^NTOTAL .*|NTOTAL                $ntotal|" -e "s|^NSTART .*|NSTART                1|" -e "s|^NSTOP .*|NSTOP                 2000|" \
      -e "s|^LSTART .*|LSTART                4812|" -e "s|^LSTOP .*|LSTOP                 $lstop|" -e "s|^RSTART .*|RSTART                1|" -e "s|^RSTOP .*|RSTOP                 4811|" \
      -e "s|^TRAJECTORY .*|TRAJECTORY            $w/prod_strip.mdcrd|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/extract_coords.mmpbsa > $w/extract_coords.mmpbsa
  sed -e "s|^PARALLEL .*|PARALLEL              8|" -e "s|^START .*|START                 1|" -e "s|^STOP .*|STOP                  2000|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/binding_energy.mmpbsa > $w/binding_energy.mmpbsa
  sed -e "s|^START .*|START                 1|" -e "s|^STOP .*|STOP                  2000|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/dec_res.mmpbsa > $w/dec_res.mmpbsa
  sed -e "s|^START .*|START                 1991|" -e "s|^STOP .*|STOP                  2000|" \
      -e "s|^COMPT .*|COMPT                 $D/md/$idx/complex.prmtop|" -e "s|^RECPT .*|RECPT                 $D/md/$idx/receptor.prmtop|" -e "s|^LIGPT .*|LIGPT                 $D/md/$idx/ligand.prmtop|" \
      $TPL/nmode.mmpbsa > $w/nmode.mmpbsa
  echo "  -> $w: $(ls $w/*.mmpbsa|wc -l) mmpbsa + prod_strip $(du -h $w/prod_strip.mdcrd 2>/dev/null|cut -f1) + rec/lig.prmtop $([ -f receptor.prmtop ]&&echo ✅)$([ -f ligand.prmtop ]&&echo ✅)"
done
echo "=== prep5 完成 ==="