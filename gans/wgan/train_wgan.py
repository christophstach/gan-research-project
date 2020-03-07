import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.logging import CometLogger

from gans.wgan import WGAN


def main(hparams):
    model = WGAN(hparams)

    use_logger = True

    if use_logger:
        comet_logger = CometLogger(
            api_key=os.environ["COMET_KEY"],
            workspace=os.environ["COMET_WORKSPACE"],  # Optional
            project_name="research-project-gan",  # Optional
            rest_api_key=os.environ["COMET_REST_KEY"],  # Optional
            experiment_name="Wasserstein GAN"  # Optional
        )
    else:
        comet_logger = None

    trainer = Trainer(
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        early_stop_callback=False,
        logger=comet_logger if use_logger else False
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--nodes', type=int, default=1)

    parser = WGAN.add_model_specific_args(parser)

    main(parser.parse_args())
