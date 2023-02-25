import argparse

from trainer import FinetuningTrainer, PreTrainer, SelfPreTrainer
from utils import str2bool


def run(args):
    trainer_dict = {"finetune": FinetuningTrainer,  "pretrain": PreTrainer, 
                    "self-pretrain": SelfPreTrainer}
    print("stage:", args.stage)
    trainer = trainer_dict[args.stage](args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument("--stage", required=True, type=str)
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)

    
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument("--norm_pix_loss", default="True", type=str2bool)
    parser.add_argument("--mask_ratio", type=float, default=0.75)

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--plot", type=str2bool, default="false")
    parser.add_argument("--model_name", type=str,
                        default="facebook/vit-mae-base")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ckpt_name", type=str)
    parser.add_argument("--scratch", type=str2bool, default="True")
    args = parser.parse_args()
    run(args)
