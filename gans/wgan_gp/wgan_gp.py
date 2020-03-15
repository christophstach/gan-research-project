import math
import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.logging import CometLogger, TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from gans.wgan_gp.models import Generator, Critic


class WGANGP(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.dataset = self.hparams.dataset

        if self.dataset == "mnist" or self.dataset == "fashion_mnist":
            self.hparams.image_channels = 1
        elif self.dataset == "cifar10":
            self.hparams.image_channels = 3

        self.loss_type = self.hparams.loss_type
        self.image_channels = self.hparams.image_channels
        self.image_size = self.hparams.image_size
        self.alternation_interval = self.hparams.alternation_interval
        self.batch_size = self.hparams.batch_size
        self.noise_size = self.hparams.noise_size
        self.y_size = self.hparams.y_size
        self.learning_rate = self.hparams.learning_rate
        self.gradient_penalty_term = self.hparams.gradient_penalty_term
        self.dataloader_num_workers = self.hparams.dataloader_num_workers
        self.weight_clipping = self.hparams.weight_clipping

        self.beta1 = self.hparams.beta1
        self.beta2 = self.hparams.beta2

        self.generator = Generator(self.hparams)
        self.critic = Critic(self.hparams)

        self.real_images = None
        self.y = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def on_train_start(self):
        if isinstance(self.logger, CometLogger):
            self.logger.experiment.set_model_graph(str(self))

    def forward(self, x, y):
        return self.generator(x, y)

    def generator_loss(self, fake_validity):
        return -fake_validity.mean()

    def critic_loss(self, real_validity, fake_validity):
        return fake_validity.mean() - real_validity.mean()

    def clip_weights(self):
        for weight in self.critic.parameters():
            weight.data.clamp_(-self.weight_clipping, self.weight_clipping)

    def gradient_penalty(self, real_images, fake_images, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn_like(real_images, device=real_images.device)

        # Get random interpolation between real and fake samples

        # interpolates = alpha * real_images + ((1 - alpha) * fake_images)
        interpolates = real_images + alpha * (fake_images - real_images)
        interpolates.requires_grad_()

        # Get gradient w.r.t. interpolates
        interpolate_validity = self.critic(interpolates, y)
        gradients = torch.autograd.grad(
            outputs=interpolate_validity, inputs=interpolates, grad_outputs=torch.ones_like(interpolate_validity), create_graph=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return penalty

    def divergence_gradient_penalty(self, real_validity, fake_validity, real_images, fake_images):
        k = 2
        p = 6

        real_images.requires_grad_()
        # fake_images.requires_grad_()

        # Compute W-div gradient penalty
        real_grad_outputs = torch.ones(real_images.size(0))
        real_gradients = torch.autograd.grad(
            outputs=real_validity, inputs=real_images, grad_outputs=real_grad_outputs, create_graph=True
        )[0]
        real_grad_norm = real_gradients.view(real_gradients.size(0), -1).pow(2).sum(1) ** (p / 2)

        fake_grad_outputs = torch.ones(fake_images.size(0))
        fake_gradients = torch.autograd.grad(
            outputs=fake_validity, inputs=fake_images, grad_outputs=fake_grad_outputs, create_graph=True, retain_graph=True
        )[0]
        fake_grad_norm = fake_gradients.view(fake_gradients.size(0), -1).pow(2).sum(1) ** (p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

        return div_gp

    def training_step_critic(self, batch):
        self.real_images, self.y = batch

        noise = torch.randn(self.real_images.size(0), self.noise_size)
        if self.on_gpu:
            noise = noise.cuda(self.real_images.device.index)

        fake_images = self.forward(noise, self.y).detach()
        real_validity = self.critic(self.real_images, self.y)
        fake_validity = self.critic(fake_images, self.y)

        if self.loss_type == "wgan-gp":
            gradient_penalty = self.gradient_penalty_term * self.gradient_penalty(self.real_images, fake_images, self.y)
        elif self.loss_type == "wgan-gp-div":
            gradient_penalty = self.divergence_gradient_penalty(real_validity, fake_validity, self.real_images, fake_images)
        else:
            gradient_penalty = 0

        loss = self.critic_loss(real_validity, fake_validity)
        logs = {"critic_loss": loss, "gradient_penalty": gradient_penalty}
        return OrderedDict({"loss": loss + gradient_penalty, "log": logs, "progress_bar": logs})

    def training_step_generator(self, batch):
        self.real_images, self.y = batch

        noise = torch.randn(self.real_images.size(0), self.noise_size)
        if self.on_gpu:
            noise = noise.cuda(self.real_images.device.index)

        fake_images = self.forward(noise, self.y)
        fake_validity = self.critic(fake_images, self.y)

        loss = self.generator_loss(fake_validity)
        logs = {"generator_loss": loss}
        return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:  # Train critic
            return self.training_step_critic(batch)

        if optimizer_idx == 1:  # Train generator
            return self.training_step_generator(batch)

    # Logs an image for each class defined as noise size
    def on_epoch_end(self):
        if self.logger:
            num_images = 16
            noise = torch.randn(num_images, self.noise_size)
            y = torch.tensor(range(num_images))

            if self.on_gpu:
                noise = noise.cuda(self.real_images.device.index)
                y = y.cuda(self.real_images.device.index)

            fake_images = self.forward(noise, y)
            grid = torchvision.utils.make_grid(fake_images, nrow=int(math.sqrt(num_images)))

            if isinstance(self.logger, TensorBoardLogger):
                # for tensorboard
                self.logger.experiment.add_image("example_images", grid, 0)
            elif isinstance(self.logger, CometLogger):
                # for comet.ml
                self.logger.experiment.log_image(
                    grid.detach().cpu().numpy(),
                    name="generated_images",
                    image_channels="first"
                )

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # update critic opt every step
        if optimizer_idx == 0:
            optimizer.step()

            if self.loss_type == "wgan-wc":
                self.clip_weights()

            optimizer.zero_grad()

        # update generator opt every {self.alternation_interval} steps
        if optimizer_idx == 1 and batch_idx % self.alternation_interval == 0:
            optimizer.step()
            optimizer.zero_grad()

    def configure_optimizers(self):
        if self.loss_type == "wgan-gp" or self.loss_type == "wgan-gp-div":
            return [
                optim.Adam(self.critic.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2)),
                optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
            ]
        elif self.loss_type == "wgan-wc":
            return [
                optim.RMSprop(self.critic.parameters(), lr=self.learning_rate),
                optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
            ]

    def prepare_data(self):
        train_resize = transforms.Resize((self.image_size, self.image_size))
        test_resize = transforms.Resize(224, 224)

        if self.image_channels == 3:
            train_normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        else:
            train_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # prepare images for the usage with torchvision models: https://pytorch.org/docs/stable/torchvision/models.html
        test_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([train_resize, transforms.ToTensor(), train_normalize])
        test_transform = transforms.Compose([test_resize, transforms.ToTensor(), test_normalize])

        if self.dataset == "mnist":
            train_set = MNIST(os.getcwd() + "/.datasets", train=True, download=True, transform=train_transform)
            test_set = MNIST(os.getcwd() + "/.datasets", train=False, download=True, transform=test_transform)
        elif self.dataset == "fashion_mnist":
            train_set = FashionMNIST(os.getcwd() + "/.datasets", train=True, download=True, transform=train_transform)
            test_set = FashionMNIST(os.getcwd() + "/.datasets", train=False, download=True, transform=test_transform)
        elif self.dataset == "cifar10":
            train_set = CIFAR10(os.getcwd() + "/.datasets", train=True, download=True, transform=train_transform)
            test_set = CIFAR10(os.getcwd() + "/.datasets", train=False, download=True, transform=test_transform)
        else:
            raise NotImplementedError("Custom dataset is not implemented yet")

        self.train_dataset = train_set
        self.test_dataset, self.val_dataset = random_split(test_set, [len(test_set) - 1000, 1000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.dataloader_num_workers, batch_size=self.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        train_group = parser.add_argument_group("Training")
        train_group.add_argument("-mine", "--min-epochs", type=int, default=1, help="Minimum number of epochs to train")
        train_group.add_argument("-maxe", "--max-epochs", type=int, default=1000, help="Maximum number of epochs to train")
        train_group.add_argument("-acb", "--accumulate-grad-batches", type=int, default=1, help="Accumulate gradient batches")
        train_group.add_argument("-dnw", "--dataloader-num-workers", type=int, default=4, help="Number of workers the dataloader uses")
        train_group.add_argument("-b1", "--beta1", type=int, default=0.5, help="Momentum term beta1")
        train_group.add_argument("-b2", "--beta2", type=int, default=0.999, help="Momentum term beta2")

        system_group = parser.add_argument_group("System")
        system_group.add_argument("-ic", "--image-channels", type=int, default=3, help="Generated image shape channels")
        system_group.add_argument("-iw", "--image-size", type=int, default=64, help="Generated image shape width")
        system_group.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size")
        system_group.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate of both optimizers")
        system_group.add_argument("-lt", "--loss-type", type=str, choices=["wgan-gp", "wgan-wc", "wgan-gp-div"], default="wgan-gp")

        system_group.add_argument("-z", "--noise-size", type=int, default=100, help="Length of the noise vector")
        system_group.add_argument("-y", "--y-size", type=int, default=10, help="Length of the y/label vector")
        system_group.add_argument("-yes", "--y-embedding-size", type=int, default=10, help="Length of the y/label embedding vector")
        system_group.add_argument("-k", "--alternation-interval", type=int, default=5, help="Amount of steps the critic is trained for each training step of the generator")

        critic_group = parser.add_argument_group("Critic")
        critic_group.add_argument("-clrs", "--critic-leaky-relu-slope", type=float, default=0.2, help="Slope of the leakyReLU activation function in the critic")
        critic_group.add_argument("-gpt", "--gradient-penalty-term", type=float, default=100, help="Gradient penalty term")
        critic_group.add_argument("-wc", "--weight-clipping", type=float, default=0.01, help="Weights of the critic gets clipped at this point")

        generator_group = parser.add_argument_group("Generator")

        return parser
