#!/bin/bash
# 编排: 等 setup 完成 -> 2 GPU 并行跑 6 体系 MD(min1->min2->heat->eq20ns->prod100ns) -> 自动轨迹分析 -> 停(不做MM-GBSA)
D=/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking
PY=/root/miniconda3/envs/pytorch/bin/python
cd $D
IDXS="2732 6218 6117 2532 5614 6046"

echo "[$(date)] === MD 编排启动 ==="
# 1. 等 6 体系 setup 完成
echo "[$(date)] 等 setup (6 体系 prmtop)..."
while true; do
  n=0; for i in $IDXS; do [ -f md/$i/complex_solvated.prmtop ] && n=$((n+1)); done
  [ $n -eq 6 ] && break
  # 若 setup 进程已死且未全就绪, 也跳出
  pgrep -f setup_md_systems.py >/dev/null || { [ $n -lt 6 ] && echo "[$(date)] setup 进程结束, ready=$n"; break; }
  sleep 60
done
echo "[$(date)] setup 就绪 ($(for i in $IDXS; do [ -f md/$i/complex_solvated.prmtop ] && echo -n "$i "; done))"

# 2. 启动 2 GPU 链 (GPU3: 2732,6218,6117 ; GPU0: 2532,5614,6046)
bash -c "for i in 2732 6218 6117; do bash $D/md/run_system.sh \$i 3; done" > logs/md_gpu3.log 2>&1 &
G3=$!
bash -c "for i in 2532 5614 6046; do bash $D/md/run_system.sh \$i 0; done" > logs/md_gpu0.log 2>&1 &
G0=$!
echo "[$(date)] 启动 MD: GPU3(PID=$G3) GPU0(PID=$G0)"
wait $G3 $G0
echo "[$(date)] === 全部 6 体系 MD 生产完成 ==="

# 3. 轨迹分析
echo "[$(date)] 跑轨迹分析 (RMSD/RMSF/Hbond)..."
$PY $D/analyze_md.py > logs/md_analysis.log 2>&1
echo "[$(date)] === 轨迹分析完成, 停步 (未做 MM-GBSA, 等用户) ==="
