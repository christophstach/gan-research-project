git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 256 --logger comet.ml --max-epochs 500 --dataset mnist
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 256 --logger comet.ml --max-epochs 500 --dataset fhasion_mnist
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 256 --logger comet.ml --max-epochs 1000 --dataset cifar10