# DTU scan65
export CUDA_VISIBLE_DEVICES=6
# python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/dtu_mlp_3views.conf  --scan_id 65

python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/studio_mlp.conf  --scan_id 1
