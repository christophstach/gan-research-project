git pull
clear

python gans/train_gan.py --gpus 0 --dataloader-num-workers 10 --batch-size 64 --logger wandb --max-epochs 5000 --dataset cifar10 --loss-strategy wgan --gradient-penalty-strategy lp --gradient-penalty-term 0.1