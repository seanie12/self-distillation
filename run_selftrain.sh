GPU=${1:-0}
DS=${2:-cifar100}
STEPS=${3:-20000}


CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --stage self-pretrain \
    --steps $STEPS \
    --warmup_steps 0 \
    --batch_size 64 \
    --weight_decay 0.01 \
    --lr 1.5e-4 \
    --norm_pix_loss True \
    --mask_ratio 0.75 \
    --workers 2 \
    --dataset $DS


