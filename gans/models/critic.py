import math
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

from ..building_blocks import DownsampleStridedConv2d, ResidualBlockTypeC


class Critic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.y_embedding = nn.Embedding(num_embeddings=self.hparams.y_size, embedding_dim=self.hparams.image_size ** 2)

        self.features = nn.Sequential(
            DownsampleStridedConv2d(self.hparams.image_channels + (1 if self.hparams.y_size > 1 else 0), self.hparams.critic_filters),
            DownsampleStridedConv2d(self.hparams.critic_filters, self.hparams.critic_filters * 2),
            DownsampleStridedConv2d(self.hparams.critic_filters * 2, self.hparams.critic_filters * 4),
            DownsampleStridedConv2d(self.hparams.critic_filters * 4, self.hparams.critic_filters * 8, kernel_size=5, stride=4, padding=False)
        )

        self.validator = nn.Sequential(
            ResidualBlockTypeC(self.hparams.critic_filters * 8, self.hparams.critic_filters, kernel_size=1),
            ResidualBlockTypeC(self.hparams.critic_filters, self.hparams.critic_filters, kernel_size=1),
            ResidualBlockTypeC(self.hparams.critic_filters, 1)
        )

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # https://arxiv.org/abs/1502.01852

            torch.nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x, y):
        if self.hparams.y_size > 1:
            y = self.y_embedding(y)
            y = y.view(y.size(0), -1, self.hparams.image_size, self.hparams.image_size)
            data = torch.cat([x, y], dim=1)
        else:
            data = x

        data = self.features(data)

        validity = self.validator(data)
        print(validity.size())
        exit(0)
        validity = validity.view(validity.size(0), -1)

        validity = validity.mean(1, keepdim=True)

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
