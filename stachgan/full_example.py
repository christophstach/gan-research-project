import os
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .generator import Generator
from .discrimator import Discriminator


class StachGAN(pl.LightningModule):
    def __init__(self, image_size=(32, 32)):
        super().__init__()

        self.image_size = image_size

        self.generator = Generator(image_shape=image_size)
        self.discriminator = Discriminator(image_shape=image_size)

    def forward(self, x):
        return self.generator.forward(x)

    def generator_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def discriminator_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = torch.randn(self.image_size)

        y_real = self.forward(x)
        y_fake = self.forward(z)

        # loss = self.generator_loss(y_hat, y)

        # logs = {"loss": loss}
        # return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.generator_loss(y_hat, y)

        # log 6 example images
        # or generated text... or whatever
        sample_imgs = x[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("example_images", grid, 0)

        # calculate acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        return OrderedDict({
            "val_loss": loss,
            "val_acc": torch.tensor(val_acc)
        })

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        logs = {"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc}

        return OrderedDict({
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "log": logs
        })

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.generator_loss(y_hat, y)

        # calculate acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        return OrderedDict({
            "test_loss": loss,
            "test_acc": torch.tensor(test_acc)
        })

    def test_end(self, outputs):
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {"avg_test_loss": avg_test_loss, "avg_test_acc": avg_test_acc}

        return OrderedDict({
            "test_loss": avg_test_loss,
            "test_acc": avg_test_acc,
            "log": logs
        })

    def configure_optimizers(self):
        return [
            optim.Adam(self.generator.parameters()),
            optim.Adam(self.discriminator.parameters())
        ]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            CIFAR10(
                os.getcwd() + "/.datasets",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=32
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            CIFAR10(
                os.getcwd() + "/.datasets",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=32
        )

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            CIFAR10(
                os.getcwd() + "/.datasets",
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=32
        )
