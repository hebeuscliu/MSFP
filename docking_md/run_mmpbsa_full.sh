#!/bin/bash
# Full MM-GBSA chain for ONE system: extract -> binding -> dec_res -> nmode
# 用法: bash run_mmpbsa_full.sh <idx>
idx=$1
AMBER=/root/disk1/senchenliu/home/amber24
export AMBERHOME=$AMBER
D=/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking
w=$D/md/$idx/energy
cd $w

echo "[$(date)] $idx step1 extract_coords"
rm -f snapshot_*.all.out snapshot_statistics* snapshot/snapshot_*.crd.*
$AMBER/bin/mm_pbsa.pl extract_coords.mmpbsa > extract.log 2>&1
n=$(ls snapshot/snapshot_com.crd.* 2>/dev/null | wc -l)
echo "[$(date)] $idx extract done ($n snapshots)"

echo "[$(date)] $idx step2 binding_energy"
rm -f snapshot_*.all.out snapshot_statistics.out
$AMBER/bin/mm_pbsa.pl binding_energy.mmpbsa > bind.log 2>&1
nc=$(grep -c '^ BOND' snapshot_com.all.out 2>/dev/null)
nr=$(grep -c '^ BOND' snapshot_rec.all.out 2>/dev/null)
echo "[$(date)] $idx binding done (com=$nc rec=$nr)"
if [ "$nc" != "$nr" ] || [ "$nc" != "2000" ]; then
  echo "[$(date)] $idx !!! binding 块数异常, 重跑"
  rm -f snapshot_*.all.out snapshot_statistics.out
  $AMBER/bin/mm_pbsa.pl binding_energy.mmpbsa > bind2.log 2>&1
fi
cp snapshot_statistics.out snapshot_statistics_binding.out 2>/dev/null
cp snapshot_com.all.out snapshot_com_binding.all.out 2>/dev/null

echo "[$(date)] $idx step3 dec_res"
rm -f snapshot_*.all.out snapshot_statistics.out
$AMBER/bin/mm_pbsa.pl dec_res.mmpbsa > dec.log 2>&1
echo "[$(date)] $idx dec_res done"
cp snapshot_statistics.out snapshot_statistics_dec.out 2>/dev/null

echo "[$(date)] $idx step4 nmode"
rm -f snapshot_*.all.out snapshot_statistics.out
$AMBER/bin/mm_pbsa.pl nmode.mmpbsa > nmode.log 2>&1
echo "[$(date)] $idx nmode done"
cp snapshot_statistics.out snapshot_statistics_nmode.out 2>/dev/null

echo "[$(date)] $idx === 全流程完成 ==="
echo "--- $idx ΔG_TOT (binding) ---"
grep -A12 "DELTA" snapshot_statistics_binding.out 2>/dev/null | grep -E "GBTOT|GBELE" | tail -2
echo "--- $idx -TΔS (nmode) ---"
grep -iE "ENTROPY|TDS|TS| -T" snapshot_statistics_nmode.out 2>/dev/null | tail -3