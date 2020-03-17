git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 5000 --dataset cifar10 --loss-type wgan-gp1