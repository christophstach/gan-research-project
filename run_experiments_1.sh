git pull
clear

python gans/train_gan.py --gpus 0 --dataloader-num-workers 10 --batch-size 64 --logger wandb --max-epochs 5000 --dataset cifar10 --loss-strategy wagan --gradient-penalty-strategy 1-gp --gradient-penalty-weight 10 --consistency-term-weight 0