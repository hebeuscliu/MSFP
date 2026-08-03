#!/bin/bash
# 单体系 MD 全链: min1 -> min2 -> heat -> eq(20ns) -> prod(100ns), 指定 GPU
# 用法: run_system.sh <idx> <gpu>
set -e
IDX=$1; GPU=$2
D=/root/disk1/senchenliu/MSFP/Final/phase25_snd1_docking
AMBER=/root/disk1/senchenliu/home/amber24
cd "$D/md/$IDX"
export AMBERHOME=$AMBER CUDA_VISIBLE_DEVICES=$GPU
PM=$AMBER/bin/pmemd.cuda
IN=$D/md/inputs
P=complex_solvated.prmtop

echo "[$(date)] idx=$IDX gpu=$GPU START"
# min1: 约束溶质最小化
$PM -O -i $IN/min1.in -o min1.out -p $P -c complex_solvated.inpcrd -r min1.rst -ref complex_solvated.inpcrd
echo "[$(date)] idx=$IDX min1 done"
# min2: 全最小化
$PM -O -i $IN/min2.in -o min2.out -p $P -c min1.rst -r min2.rst
echo "[$(date)] idx=$IDX min2 done"
# heat: 0->300K NVT 50ps
$PM -O -i $IN/heat.in -o heat.out -p $P -c min2.rst -r heat.rst -x heat.mdcrd -ref min2.rst
echo "[$(date)] idx=$IDX heat done"
# eq: NPT 20ns
$PM -O -i $IN/eq.in -o eq.out -p $P -c heat.rst -r eq.rst -x eq.mdcrd
echo "[$(date)] idx=$IDX eq(20ns) done"
# prod: NPT 100ns
$PM -O -i $IN/prod.in -o prod.out -p $P -c eq.rst -r prod.rst -x prod.mdcrd
echo "[$(date)] idx=$IDX PROD(100ns) DONE"
