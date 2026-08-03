#!/bin/bash
# Chain: 等 3 体系 dec_res 全完 -> 备份 dec 统计 -> 并行启动 3 体系 nmode -> 汇报
AMBER=/root/disk1/senchenliu/home/amber24
export AMBERHOME=$AMBER
D=/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking

echo "[$(date)] 等待 3 体系 dec_res 完成 (每60s轮询)..."
while pgrep -f "mm_pbsa.pl dec_res" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date)] dec_res 全部完成"

for idx in 5614 6115 2732; do
  w=$D/md/$idx/energy
  cd $w
  cp snapshot_statistics.out snapshot_statistics_dec.out 2>/dev/null
  rm -f snapshot_*.all.out snapshot_statistics.out
  nohup $AMBER/bin/mm_pbsa.pl nmode.mmpbsa > nmode_run.log 2>&1 &
  echo "[$(date)] $idx nmode 启动 PID=$!"
done

echo "[$(date)] === 3 体系 nmode 并行启动, 等待完成 (~5h) ==="
wait
echo "[$(date)] === nmode 全部完成 ==="

for idx in 5614 6115 2732; do
  echo "--- $idx -TΔS (entropy) ---"
  grep -iE "ENTROPY|TDS|TS|DELTA" $D/md/$idx/energy/snapshot_statistics.out 2>/dev/null | tail -8
done
echo "[$(date)] === MM-GBSA 全流程结束 (binding+dec+nmode) ==="