# DTU scan65
ulimit -Su unlimited
ulimit -Sv unlimited
ulimit -Sx unlimited

export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-11.3/

source /localscratch/smh31/envs/miniconda/etc/profile.d/conda.sh
conda activate monosdf

# python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/dtu_mlp_3views.conf  --scan_id 65

python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/studio_mlp.conf --scan_id 1 --is_continue 
# --is_continue --timestamp 2024_02_01_13_35_10
