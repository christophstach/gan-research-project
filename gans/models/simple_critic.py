from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score


class SimpleCritic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.hparams.image_channels, self.hparams.critic_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters) x 32 x 32
            nn.Conv2d(self.hparams.critic_filters, self.hparams.critic_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.critic_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters*2) x 16 x 16
            nn.Conv2d(self.hparams.critic_filters * 2, self.hparams.critic_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.critic_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters*4) x 8 x 8
            nn.Conv2d(self.hparams.critic_filters * 4, self.hparams.critic_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.critic_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters*8) x 4 x 4
            nn.Conv2d(self.hparams.critic_filters * 8, 1, 4, 1, 0, bias=False),
        )

        self.hparams = hparams

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, y):
        validity = self.main(x)

        return validity

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
        acc = torch.tensor(accuracy_score(y.cpu(), prediction.cpu().argmax(dim=1)), device=loss.device)

        return OrderedDict({"val_loss": loss, "val_acc": acc})

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()

        logs = {"val_loss_mean": val_loss_mean, "val_acc_mean": val_acc_mean}
        return OrderedDict({"val_loss": val_loss_mean, "val_acc_mean": val_acc_mean, "progress_bar": logs})

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.critic_learning_rate, betas=(self.hparams.critic_beta1, self.hparams.critic_beta2)),
