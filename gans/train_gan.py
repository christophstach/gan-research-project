import os
from argparse import ArgumentParser

import wandb
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import CometLogger, TensorBoardLogger, WandbLogger

from gans.applications import GAN
from gans.models import Generator, Discriminator

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    generator = Generator(hparams)
    discriminator = Discriminator(hparams)

    # scorer = models.mobilenet_v2(pretrained=True)
    model = GAN(hparams, generator, discriminator)

    experiment_name = hparams.loss_strategy + "+" + hparams.architecture
    if hparams.gradient_penalty_strategy != "none":
        experiment_name += "+" + hparams.gradient_penalty_strategy
    if hparams.multi_scale_gradient:
        experiment_name += "+msg"
    if hparams.instance_noise:
        experiment_name += "+in"
    if hparams.spectral_normalization:
        experiment_name += "+sn"
    if hparams.equalized_learning_rate:
        experiment_name += "+eqlr"

    experiment_name += " (" + hparams.dataset + ")"

    if hparams.logger == "none":
        logger = False
    elif hparams.logger == "comet.ml":
        logger = CometLogger(
            api_key=os.environ["COMET_KEY"],
            workspace=os.environ["COMET_WORKSPACE"],  # Optional
            project_name="gan-research-project",  # Optional
            rest_api_key=os.environ["COMET_REST_KEY"],  # Optional
            experiment_name=experiment_name
        )
    elif hparams.logger == "wandb":
        logger = WandbLogger(
            project="gan-research-project",
            name=experiment_name
        )

        logger.watch(model)
    elif hparams.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd() + "/lightning_logs"
        )
    else:
        raise ValueError("Must specific a logger")

    if hparams.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.getcwd() + "/checkpoints/{epoch}-" + hparams.loss_strategy + "+" + hparams.gradient_penalty_strategy + "+" + hparams.dataset + "-{discriminator_loss:.5f}-{ic_score_mean:.5f}",
            monitor="discriminator_loss",
            mode="max",
            save_top_k=1,
            period=1
        )
    else:
        checkpoint_callback = False

    trainer = Trainer(
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        progress_bar_refresh_rate=20,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        fast_dev_run=False,
        num_sanity_val_steps=0,
        distributed_backend="dp",
        weights_summary=None
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--logger", type=str, choices=["none", "comet.ml", "tensorboard", "wandb"], required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--save-checkpoints", action="store_true")

    parser = GAN.add_model_specific_args(parser)

    hparams = parser.parse_args()

    if hparams.dataset == "mnist" or hparams.dataset == "fashion_mnist":
        hparams.image_channels = 1
    elif hparams.dataset == "cifar10":
        hparams.image_channels = 3

    main(hparams)
