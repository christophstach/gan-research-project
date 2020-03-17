git pull
clear

python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 500 --dataset fashion_mnist --loss-type wgan-gp1
# python gans/wgan_gp/train_wgan_gp.py --gpus 1 --batch-size 64 --logger comet.ml --max-epochs 50 --dataset mnist --loss-type wgan-gp2