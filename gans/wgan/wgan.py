import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from gans.wgan.models import SimpleGenerator, SimpleDiscriminator


class WGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.alternation_interval = self.hparams.alternation_interval
        self.batch_size = self.hparams.batch_size
        self.noise_size = self.hparams.noise_size
        self.learning_rate = self.hparams.learning_rate
        self.beta1 = self.hparams.beta1

        self.generator = SimpleGenerator(self.hparams)
        self.discriminator = SimpleDiscriminator(self.hparams)

        # image cache so both optimizer optimize on the same images
        self.real_images = None
        self.fake_images = None

    def forward(self, x):
        return self.generator.forward(x)

    def adversial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def generator_loss(self, fake_images):
        fake_labels = torch.ones(fake_images.shape[0])

        if self.on_gpu:
            fake_labels = fake_labels.cuda(fake_images.device.index)

        fake_images = self.discriminator(fake_images).view(-1)

        return self.adversial_loss(fake_images, fake_labels)

    def discriminator_loss(self, real_images, fake_images):
        real_labels = torch.ones(real_images.shape[0])
        fake_labels = torch.zeros(fake_images.shape[0])

        if self.on_gpu:
            real_labels = real_labels.cuda(real_images.device.index)
            fake_labels = fake_labels.cuda(fake_images.device.index)

        real_loss = self.adversial_loss(self.discriminator(real_images).view(-1), real_labels)
        fake_loss = self.adversial_loss(self.discriminator(fake_images).view(-1), fake_labels)

        return (real_loss + fake_loss) / 2

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch
        self.real_images = real_images

        if optimizer_idx == 0:  # Train generator
            noise = torch.randn(real_images.shape[0], self.noise_size, 1, 1)
            if self.on_gpu:
                noise = noise.cuda(real_images.device.index)

            self.fake_images = self.generator(noise)

            loss = self.generator_loss(self.fake_images)
            logs = {"generator_loss": loss}
            return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})

        if optimizer_idx == 1:  # Train discriminator
            loss = 0

            for _ in range(self.alternation_interval):
                loss += self.discriminator_loss(self.real_images, self.fake_images.detach())

            loss /= self.alternation_interval
            logs = {"discriminator_loss": loss}
            return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})

    def configure_optimizers(self):
        return [
            optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)),
            optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
        ]

    def on_epoch_end(self):
        self.generator.eval()
        grid = torchvision.utils.make_grid(self.fake_images[:6], nrow=3)

        # for tensorboard
        # self.logger.experiment.add_image("example_images", grid, 0)

        # for comet.ml
        self.logger.experiment.log_image(
            grid.detach().cpu().numpy(),
            name="Fake Images",
            image_channels="first"
        )

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            MNIST(
                os.getcwd() + "/.datasets",
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((self.image_width, self.image_height)),
                    transforms.ToTensor(),
                ])
            ),
            batch_size=self.batch_size
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        train_group = parser.add_argument_group("Training")
        train_group.add_argument("-mine", "--min_nb_epochs", type=int, default=1, help="Minimum number of epochs to train")
        train_group.add_argument("-mane", "--max_nb_epochs", type=int, default=20, help="Maximum number of epochs to train")
        train_group.add_argument("-acb", "--accumulate_grad_batches", type=int, default=1, help="Accumulate gradient batches")

        system_group = parser.add_argument_group("System")
        system_group.add_argument("-ic", "--image-channels", type=int, default=1, help="Generated image shape channels")
        system_group.add_argument("-iw", "--image-width", type=int, default=64, help="Generated image shape width")
        system_group.add_argument("-ih", "--image-height", type=int, default=64, help="Generated image shape height")
        system_group.add_argument("-bs", "--batch-size", type=int, default=32, help="Batch size")
        system_group.add_argument("-lr", "--learning-rate", type=float, default=2e-4, help="Learning rate of both optimizers")
        system_group.add_argument("-z", "--noise-size", type=int, default=100, help="Length of the noise vector")
        system_group.add_argument("-k", "--alternation-interval", type=int, default=1, help="Amount of steps the discriminator for each training step of the generator")
        system_group.add_argument("-b1", "--beta1", type=float, default=0.5, help="Momentum term beta1")

        discriminator_group = parser.add_argument_group("Discriminator")
        discriminator_group.add_argument("-dlrs", "--discriminator-leaky-relu-slope", type=float, default=0.2, help="Slope of the leakyReLU activation function in the discriminator")
        discriminator_group.add_argument("-df", "--discriminator-filters", type=int, default=64, help="Filters in the discriminator (are multiplied with different powers of 2)")
        discriminator_group.add_argument("-dl", "--discriminator-length", type=int, default=3, help="Length of the discriminator or number of down sampling blocks")

        generator_group = parser.add_argument_group("Generator")
        generator_group.add_argument("-gf", "--generator-filters", type=int, default=32, help="Filters in the generator (are multiplied with different powers of 2)")
        generator_group.add_argument("-gl", "--generator-length", type=int, default=3, help="Length of the generator or number of up sampling blocks (also determines the size of the output image)")

        return parser
