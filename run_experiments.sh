git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 20 --dataset mnist --loss-type wgan-gp1
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 20 --dataset mnist --loss-type wgan-gp2
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 20 --dataset fashion_mnist --loss-type wgan-gp1
python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 20 --dataset cifar10 --loss-type wgan-gp1