import os
from argparse import ArgumentParser

import torchvision.models as models
from pytorch_lightning import Trainer
from pytorch_lightning.logging import CometLogger, TensorBoardLogger, WandbLogger

from gans.wgan_gp import WGANGP
from gans.wgan_gp.models import Generator, Critic


def main(hparams):
    generator = Generator(hparams)
    critic = Critic(hparams)
    scorer = models.inception_v3(pretrained=True)
    model = WGANGP(hparams, generator, critic, scorer)

    if hparams.logger == "none":
        logger = False
    elif hparams.logger == "comet.ml":
        logger = CometLogger(
            api_key=os.environ["COMET_KEY"],
            workspace=os.environ["COMET_WORKSPACE"],  # Optional
            project_name="research-project-gan",  # Optional
            rest_api_key=os.environ["COMET_REST_KEY"],  # Optional
            experiment_name="Wasserstein GAN+GP (" + hparams.dataset + ")"  # Optional
        )
    elif hparams.logger == "wandb":
        logger = WandbLogger(
            project="research-project-gan",
            name="Wasserstein GAN+GP (" + hparams.dataset + ")"
        )
    elif hparams.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd() + "/lightning_logs"
        )
    else:
        raise ValueError("Must specific a logger")

    trainer = Trainer(
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        progress_bar_refresh_rate=20,
        early_stop_callback=False,
        checkpoint_callback=False,
        logger=logger,
        fast_dev_run=False,
        num_sanity_val_steps=0
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--logger", type=str, choices=["none", "comet.ml", "tensorboard"], required=True)
    parser.add_argument("--dataset", type=str, choices=["custom", "cifar10", "mnist", "fashion_mnist", "wandb"], required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=0)
    parser.add_argument("--nodes", type=int, default=1)

    parser = WGANGP.add_model_specific_args(parser)

    hparams = parser.parse_args()

    if hparams.dataset == "mnist" or hparams.dataset == "fashion_mnist":
        hparams.image_channels = 1
    elif hparams.dataset == "cifar10":
        hparams.image_channels = 3

    main(hparams)
