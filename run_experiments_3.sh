python gans/train_gan.py \
  --gpus 1 \
  --max-epochs 10000 \
  --dataset celeba_hq \
  --dataloader-num-workers 10 \
  --exponential-filter-multipliers \
  --generator-filters 4 \
  --discriminator-filters 4 \
  --batch-size 32 \
  --image-size 256 \
  --noise-size 128 \
  --logger wandb \
  --loss-strategy ra-hinge \
  --architecture hdcgan \
  --multi-scale-gradient \
  --spectral-normalization \
  --instance-noise