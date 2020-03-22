git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 0 1 --dataloader-num-workers 10 --batch-size 128 --logger wandb --max-epochs 5000 --dataset cifar10 --loss-type wgan-gp1