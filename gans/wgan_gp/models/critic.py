from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score


class Critic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.pretrain = False

        self.main = nn.Sequential(
            # input is (self.hparams.image_channels) x 64 x 64
            nn.Conv2d(self.hparams.image_channels, self.hparams.image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.hparams.image_size) x 32 x 32
            nn.Conv2d(self.hparams.image_size, self.hparams.image_size * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.hparams.image_size * 2),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.hparams.image_size*2) x 16 x 16
            nn.Conv2d(self.hparams.image_size * 2, self.hparams.image_size * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.hparams.image_size * 4),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.hparams.image_size*4) x 8 x 8
            nn.Conv2d(self.hparams.image_size * 4, self.hparams.image_size * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.hparams.image_size * 8),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.hparams.image_size*8) x 4 x 4

        )

        self.validator = nn.Sequential(
            nn.Conv2d(self.hparams.image_size * 8, 1, 4, 1, 0, bias=False)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(self.hparams.image_size * 8, self.hparams.y_size, 4, 1, 0, bias=False)
        )

    def forward(self, x, y):
        x = self.main(x)

        if self.pretrain:
            x = self.classifier(x)
            x = x.squeeze()
            x = torch.sigmoid(x)
        else:
            x = self.validator(x)
            x = x.squeeze()

        return x

    """Below methods are just used for pretraining the critic"""

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.forward(x, y)

        loss = F.cross_entropy(prediction, y)
        return OrderedDict({"loss": loss})

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.forward(x, y)

        loss = F.cross_entropy(prediction, y)
        acc = accuracy_score(y.cpu(), prediction.cpu().argmax(dim=1))

        logs = {"val_acc": acc}
        return OrderedDict({"val_loss": loss, "val_acc": acc, "logs": logs})

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        logs = {"val_loss": val_loss_mean, "val_acc_mean": val_acc_mean}
        return {"val_loss": val_loss_mean, "val_acc_mean": val_acc_mean, "progress_bar": logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta1, self.hparams.beta2)),
