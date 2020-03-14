import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.logging import CometLogger, TensorBoardLogger

from gans.wgan_gp import WGANGP


def main(hparams):
    model = WGANGP(hparams)

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
    elif hparams.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd() + "/lightning_logs",
        )
    else:
        raise ValueError("Must specific a logger")

    trainer = Trainer(
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        progress_bar_refresh_rate=1,
        early_stop_callback=False,
        checkpoint_callback=False,
        logger=logger
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--logger", type=str, choices=["none", "comet.ml", "tensorboard"], required=True)
    parser.add_argument("--dataset", type=str, choices=["custom", "cifar10", "mnist", "fashion_mnist"], required=True)
    parser.add_argument("--gpus", type=int, default=0   )
    parser.add_argument("--nodes", type=int, default=1)

    parser = WGANGP.add_model_specific_args(parser)

    main(parser.parse_args())
