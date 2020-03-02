import os

from collections import OrderedDict
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from .discrimator import Discriminator
from .generator import Generator



from PIL import Image


class StachGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.alternation_interval = self.hparams.alternation_interval
        self.batch_size = self.hparams.batch_size
        self.noise = self.hparams.noise
        self.learning_rate = self.hparams.learning_rate
        self.beta1 = self.hparams.beta1

        self.generator = Generator(self.hparams)
        self.discriminator = Discriminator(self.hparams)
        self.adversial_loss = nn.BCELoss()

    def forward(self, x):
        return self.generator.forward(x)

    def generator_loss(self, fake_images):
        fake = torch.ones(fake_images.shape[0], 1)

        if self.on_gpu:
            fake = fake.cuda(fake_images.device.index)

        fake_images = self.discriminator(fake_images)
        return self.adversial_loss(fake_images, fake)

    def discriminator_loss(self, real_images, fake_images):
        real = torch.ones(real_images.shape[0], 1)
        fake = torch.zeros(fake_images.shape[0], 1)

        if self.on_gpu:
            real = real.cuda(real_images.device.index)
            fake = fake.cuda(fake_images.device.index)

        return (self.adversial_loss(self.discriminator(real_images), real)
                + self.adversial_loss(self.discriminator(fake_images), fake)) / 2

    # def generator_loss(self, fake_images):
    #     fake = torch.ones(fake_images.shape[0])

    #     if self.on_gpu:
    #         fake = fake.cuda(fake_images.device.index)

    #     fake_images = self.discriminator(fake_images).view(-1)

    #     return (fake - fake_images).mean()


    # def discriminator_loss(self, real_images, fake_images):
    #     real = torch.ones(real_images.shape[0])
    #     # fake = torch.zeros(fake_images.shape[0])

    #     if self.on_gpu:
    #         real = real.cuda(real_images.device.index)
    #         # fake = fake.cuda(fake_images.device.index)

    #     real_images = self.discriminator(real_images).view(-1)
    #     fake_images = self.discriminator(fake_images).view(-1)

    #     return (real - real_images + fake_images).mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch

        if optimizer_idx == 0:  # Train discriminator
            loss = 0

            for i in range(self.alternation_interval):
                self.discriminator.train()
                self.generator.eval()

                noise = torch.randn(real_images.shape[0], self.noise, 1, 1)
                if self.on_gpu:
                    noise = noise.cuda(real_images.device.index)

                fake_images = self.generator(noise).detach()

                loss += self.discriminator_loss(real_images, fake_images)

            loss /= self.alternation_interval
            logs = {"discriminator_loss": loss}
            return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})
        
        if optimizer_idx == 1:  # Train generator
            self.discriminator.eval()
            self.generator.train()

            noise = torch.randn(real_images.shape[0], self.noise, 1, 1)
            if self.on_gpu:
                noise = noise.cuda(real_images.device.index)

            fake_images = self.generator(noise)
            loss = self.generator_loss(fake_images)

            if batch_idx % 50 == 0:
                grid = torchvision.utils.make_grid(fake_images[:6])
                
                # for tensorboard
                self.logger.experiment.add_image("example_images", grid, 0)
                # for others
                # self.logger.experiment.log_image(grid[0].detach().cpu().numpy())

            logs = {"generator_loss": loss}
            return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})

    def configure_optimizers(self):
        return [
            optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)),
            optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)),
        ], []

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
        system_group.add_argument("-iw", "--image-width", type=int, default=32, help="Generated image shape width")
        system_group.add_argument("-ih", "--image-height", type=int, default=32, help="Generated image shape height")
        system_group.add_argument("-bs", "--batch-size", type=int, default=32, help="Batch size")
        system_group.add_argument("-lr", "--learning-rate", type=float, default=2e-4, help="Learning rate of both optimizers")
        system_group.add_argument("-z", "--noise", type=int, default=100, help="Length of the noise vector")
        system_group.add_argument("-k", "--alternation-interval", type=int, default=1, help="Amount of steps the discriminator for each training step of the generator")
        system_group.add_argument("-b1", "--beta1", type=float, default=0.5, help="Momentum term beta1")

        discriminator_group = parser.add_argument_group("Discriminator")
        discriminator_group.add_argument("-dlrs", "--discriminator-leaky-relu-slope", type=float, default=0.2, help="Slope of the leakyReLU activation function in the discriminator")
        discriminator_group.add_argument("-ddr", "--discriminator-dropout-rate", type=float, default=0.2, help="Dropout rate in the discriminator")
        discriminator_group.add_argument("-df", "--discriminator-filters", type=int, default=2, help="Filters in the discriminator (are multiplied with different powers of 2)")
        discriminator_group.add_argument("-dld", "--discriminator-latent-dim", type=int, default=256, help="Size of the latent dimensions in the generator (are multiplied with different powers of 2)")

        generator_group = parser.add_argument_group("Generator")
        generator_group.add_argument("-gf", "--generator-filters", type=int, default=4, help="Filters in the generator (are multiplied with different powers of 2)")
        generator_group.add_argument("-gl", "--generator-length", type=int, default=2, help="Length of the generator or number of up sampling blocks (also determines the size of the output image)")

        return parser
