from argparse import ArgumentParser

from pytorch_lightning import Trainer

from stachgan import StachGAN


def main(hparams):
    model = StachGAN(hparams)

    trainer = Trainer(
        min_nb_epochs=hparams.min_nb_epochs,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        early_stop_callback=False
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--nodes', type=int, default=1)

    parser = StachGAN.add_model_specific_args(parser)

    main(parser.parse_args())
