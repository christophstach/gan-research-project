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

        self.features = nn.Sequential(

            nn.Conv2d(self.hparams.image_channels, self.hparams.image_size, 4, 2, 1),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(self.hparams.image_size, self.hparams.image_size * 2, 4, 2, 1),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(self.hparams.image_size * 2, self.hparams.image_size * 4, 4, 2, 1),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            # nn.Conv2d(self.hparams.image_size * 4, self.hparams.image_size * 8, 4, 2, 1),
            # nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.hparams.image_size*8) x 4 x 4

        )

        self.y_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self.hparams.y_size, embedding_dim=self.hparams.y_embedding_size * 16)
        )

        self.validator = nn.Sequential(
            nn.Conv2d(self.hparams.y_embedding_size + self.hparams.image_size * 4, 1024, 4, 1, 0),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(512, 256, 1, 1, 0),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(256, 1, 1, 1, 0),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(self.hparams.image_size * 4, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),

            nn.Conv2d(256, self.hparams.y_size, 1, 1, 0),
        )

    def forward(self, x, y):
        x = self.features(x)

        if self.pretrain:
            x = self.classifier(x)
            x = x.squeeze()
            x = torch.sigmoid(x)
        else:
            y = self.y_embedding(y)
            y = y.view(y.size(0), -1, 4, 4)
            x = torch.cat([x, y], dim=1)
            # Comment lines above to disable conditional gan

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
        acc = torch.tensor(accuracy_score(y.cpu(), prediction.cpu().argmax(dim=1)), device=loss.device)

        return OrderedDict({"val_loss": loss, "val_acc": acc})

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()

        logs = {"val_loss_mean": val_loss_mean, "val_acc_mean": val_acc_mean}
        return OrderedDict({"val_loss": val_loss_mean, "val_acc_mean": val_acc_mean, "progress_bar": logs})

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta1, self.hparams.beta2)),
