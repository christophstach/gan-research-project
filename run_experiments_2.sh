python gans/train_gan.py --gpus 1 --dataloader-num-workers 10 --batch-size 64 --logger wandb --max-epochs 10000 --dataset lsun --loss-strategy ns --gradient-penalty-strategy lp --multi-scale-gradient --save-checkpoints