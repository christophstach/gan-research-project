import math
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score


class PretrainableCritic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.pretrain = False

        self.features = nn.Sequential(
            nn.Conv2d(self.hparams.image_channels, self.hparams.critic_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.hparams.image_size, self.hparams.critic_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.hparams.image_size * 2, self.hparams.critic_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.y_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self.hparams.y_size, embedding_dim=self.hparams.y_embedding_size * 16)
        )

        self.validator = nn.Sequential(
            nn.Conv2d(self.hparams.y_embedding_size + self.hparams.image_size * 4, 512, kernel_size=4, stride=1),
            # nn.LayerNorm([512, 1, 1]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            # nn.LayerNorm([256, 1, 1]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            # nn.LayerNorm([128, 1, 1]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, kernel_size=1, stride=1),
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
        x = self.features(x)

        if self.pretrain:
            x = self.classifier(x)
            x = x.squeeze()
            x = torch.sigmoid(x)
        else:
            y = self.y_embedding(y)
            y = y.view(y.size(0), -1, 4, 4)
            x = torch.cat([x, y], dim=1)
            # Comment lines above to disable conditional gans

            x = self.validator(x)
            x = x.view(x.size(0), -1)

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
