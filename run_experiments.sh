git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 200 --dataset mnist
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 200 --dataset fashion_mnist
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 200 --dataset cifar10
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 2000 --dataset cifar10