export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

torchrun --standalone --nproc_per_node=gpu --nnodes=1 train.py
