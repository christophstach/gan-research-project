import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import CometLogger

from gans.dcgan import DCGAN


def main(hparams):
    model = DCGAN(hparams)

    comet_logger = CometLogger(
        api_key=os.environ["COMET_KEY"],
        workspace=os.environ["COMET_WORKSPACE"],  # Optional
        project_name="research-project-gan",  # Optional
        rest_api_key=os.environ["COMET_REST_KEY"],  # Optional
        experiment_name="default"  # Optional
    )

    trainer = Trainer(
        min_nb_epochs=hparams.min_nb_epochs,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        early_stop_callback=False,
        logger=comet_logger
    )

    trainer.fit(model)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--nodes', type=int, default=1)

    parser = DCGAN.add_model_specific_args(parser)

    main(parser.parse_args())
