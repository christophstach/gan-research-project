git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 1 --dataloader-num-workers 10 --batch-size 64 --logger wandb --max-epochs 5000 --dataset cifar10 --loss-type wgan-gp2