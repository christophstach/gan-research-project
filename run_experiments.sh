git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 50 --dataset mnist
# python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 50 --dataset fashion_mnist
# python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 50 --dataset cifar10