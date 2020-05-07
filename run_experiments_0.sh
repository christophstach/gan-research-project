python gans/train_gan.py \
  --gpus 0 \
  --max-epochs 10000 \
  --dataset lsun \
  --dataloader-num-workers 10 \
  --batch-size 128 \
  --image-size 128 \
  --logger wandb \
  --loss-strategy ra-sgan \
  --architecture progan \
  --weight-init he \
  --multi-scale-gradient \
  --instance-noise