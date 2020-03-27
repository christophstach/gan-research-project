import os
from argparse import ArgumentParser

import torchvision.models as models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import CometLogger, TensorBoardLogger, WandbLogger

from gans.applications import GAN
from gans.models import Generator, Critic, SimpleCritic, SimpleGenerator, SimpleSpectralNormGenerator, SimpleSpectralNormCritic


def main(hparams):
    generator = SimpleGenerator(hparams)
    critic = SimpleSpectralNormCritic(hparams)
    scorer = models.mobilenet_v2(pretrained=True)
    model = GAN(hparams, generator, critic, scorer)

    if hparams.logger == "none":
        logger = False
    elif hparams.logger == "comet.ml":
        logger = CometLogger(
            api_key=os.environ["COMET_KEY"],
            workspace=os.environ["COMET_WORKSPACE"],  # Optional
            project_name="gan-research-project",  # Optional
            rest_api_key=os.environ["COMET_REST_KEY"],  # Optional
            experiment_name=hparams.strategy + " (" + hparams.dataset + ")"  # Optional
        )
    elif hparams.logger == "wandb":
        logger = WandbLogger(
            project="gan-research-project",
            name=hparams.strategy + " (" + hparams.dataset + ")"  # Optional
        )
    elif hparams.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd() + "/lightning_logs"
        )
    else:
        raise ValueError("Must specific a logger")

    if hparams.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.getcwd() + "/checkpoints/{epoch}-" + hparams.strategy + "-{negative_critic_loss:.5f}",
            monitor="critic_loss",
            mode="max",
            save_top_k=10,
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
        distributed_backend="dp"
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--logger", type=str, choices=["none", "comet.ml", "tensorboard", "wandb"], required=True)
    parser.add_argument("--dataset", type=str, choices=["custom", "cifar10", "mnist", "fashion_mnist", "wandb"], required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--save-checkpoints", type=bool, default=False)

    parser = GAN.add_model_specific_args(parser)

    hparams = parser.parse_args()

    if hparams.dataset == "mnist" or hparams.dataset == "fashion_mnist":
        hparams.image_channels = 1
    elif hparams.dataset == "cifar10":
        hparams.image_channels = 3

    main(hparams)
