GPU=${1:-0}
DS=${2:-cub}
SCRATCH=${3:-False}
CKPT=${4:-self-further-pretrain-20000}
CKPT_DIR=${5:-checkpoints}
for i in 1 2 3 4 5
do 
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --stage finetune \
        --steps 10000 \
        --batch_size 32 \
        --weight_decay 0.01 \
        --warmup_steps 500 \
        --lr 1e-4 \
        --dataset $DS \
        --gpu $GPU \
        --ckpt_name $CKPT \
        --scratch $SCRATCH \
        --plot True \
        --checkpoint_dir $CKPT_DIR
done