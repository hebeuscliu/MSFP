#!/bin/bash
# Batch2: 6 新候选 (543,5743,6115,3188,6754,5738) MD, 3 GPU 并行(每GPU 2体系串行) -> 轨迹分析(全12) -> 停步(不做MM-GBSA)
D=/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking
PY=/root/miniconda3/envs/pytorch/bin/python
cd $D
IDXS="543 5743 6115 3188 6754 5738"

echo "[$(date)] === Batch2 MD 编排启动 ==="
# 1. 制备 6 新体系 (setup 自动跳过已完成; 每个重对接ex=32+antechamber+tleap)
echo "[$(date)] 制备 6 新体系 (重对接ex=32 + antechamber + tleap)..."
$PY $D/setup_md_systems.py > logs/setup_batch2.log 2>&1
echo "[$(date)] setup 退出码=$?, 检查 prmtop..."
n=0; for i in $IDXS; do [ -f md/$i/complex_solvated.prmtop ] && n=$((n+1)); done
echo "[$(date)] ready=$n/6"
if [ $n -lt 6 ]; then
  echo "[$(date)] !!! 制备不全 ($n/6), 缺失体系:"
  for i in $IDXS; do [ -f md/$i/complex_solvated.prmtop ] || echo "    $i"; done
  echo "[$(date)] 查看 logs/setup_batch2.log 排查, 退出"
  exit 1
fi

# 2. 3 GPU 链 (GPU3最闲: 543,5743 ; GPU7次: 6115,3188 ; GPU0有占但可用: 6754,5738)
bash -c "for i in 543 5743; do bash $D/md/run_system.sh \$i 3; done" > logs/md_b2_gpu3.log 2>&1 &
G3=$!
bash -c "for i in 6115 3188; do bash $D/md/run_system.sh \$i 7; done" > logs/md_b2_gpu7.log 2>&1 &
G7=$!
bash -c "for i in 6754 5738; do bash $D/md/run_system.sh \$i 0; done" > logs/md_b2_gpu0.log 2>&1 &
G0=$!
echo "[$(date)] 启动 Batch2 MD: GPU3(PID=$G3)=543,5743  GPU7(PID=$G7)=6115,3188  GPU0(PID=$G0)=6754,5738"
wait $G0 $G3 $G7
echo "[$(date)] === Batch2 6 体系 MD 生产完成 ==="

# 3. 轨迹分析 (全 12 体系, analyze_md.py IDXS 已含全12)
echo "[$(date)] 跑轨迹分析 (全12体系)..."
$PY $D/analyze_md.py > logs/md_analysis_b2.log 2>&1
echo "[$(date)] === Batch2 完成, 停步 (未做 MM-GBSA, 等用户) ==="
