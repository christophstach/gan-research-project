import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
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

        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.alternation_interval = self.hparams.alternation_interval
        self.batch_size = self.hparams.batch_size
        self.noise_size = self.hparams.noise_size
        self.y_size = self.hparams.y_size
        self.learning_rate = self.hparams.learning_rate
        self.gradient_penalty_term = self.hparams.gradient_penalty_term
        self.dataloader_num_workers = self.hparams.dataloader_num_workers

        self.beta1 = self.hparams.beta1
        self.beta2 = self.hparams.beta2

        self.generator = Generator(self.hparams)
        self.critic = Critic(self.hparams)

        self.real_images = None
        self.fake_images = None
        self.y = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.inception_model = None

    def forward(self, x, y):
        return self.generator.forward(x, y)

    def generator_loss(self, fake_images, y):
        return -torch.mean(self.critic(fake_images, y))

    def critic_loss(self, real_images, fake_images, y):
        return -torch.mean(self.critic(real_images, y)) + torch.mean(self.critic(fake_images, y))

    def gradient_penalty(self, real_images, fake_images, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        epsilon = torch.randn(real_images.size(0), 1, 1, 1, requires_grad=True)
        grad_outputs = torch.ones(real_images.size(0), requires_grad=True)

        if self.on_gpu:
            epsilon = epsilon.cuda(self.real_images.device.index)
            grad_outputs = grad_outputs.cuda(self.real_images.device.index)

        # Get random interpolation between real and fake samples
        interpolates = epsilon * real_images + ((1 - epsilon) * fake_images)

        # Get gradient w.r.t. interpolates
        gradients, = torch.autograd.grad(outputs=self.critic(interpolates, y), inputs=interpolates, grad_outputs=grad_outputs, create_graph=True)
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.real_images, self.y = batch

        if optimizer_idx == 0:  # Train generator
            noise = torch.randn(self.real_images.size(0), self.noise_size, 1, 1)
            if self.on_gpu:
                noise = noise.cuda(self.real_images.device.index)

            self.fake_images = self.generator(noise, self.y)

            loss = self.generator_loss(self.fake_images, self.y)
            logs = {"loss": loss, "generator_loss": loss}
            return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})

        if optimizer_idx == 1:  # Train critic
            noise = torch.randn(self.real_images.size(0), self.noise_size, 1, 1)
            if self.on_gpu:
                noise = noise.cuda(self.real_images.device.index)

            self.fake_images = self.generator(noise, self.y).detach()

            gradient_penalty = self.gradient_penalty_term * self.gradient_penalty(self.real_images, self.fake_images, self.y)
            loss = self.critic_loss(self.real_images, self.fake_images, self.y)
            logs = {"loss": loss, "critic_loss": loss, "gradient_penalty": gradient_penalty}
            return OrderedDict({"loss": loss + gradient_penalty, "log": logs, "progress_bar": logs})

    # def validation_step(self, batch, batch_idx):

    # real_images, y = batch

    # noise = torch.randn(real_images.size(0), self.noise_size, 1, 1)
    # if self.on_gpu:
    # noise = noise.cuda(self.real_images.device.index)

    # fake_images = self.generator(noise, y)

    # prediction = self.inception_model(fake_images)

    # return OrderedDict({})

    # def validation_epoch_end(self, outputs: list):
    # logs = {}
    # return {"log": logs}

    # Logs an image for each class defined as noise size
    def on_epoch_end(self):
        if self.logger:
            num_images = self.y_size if self.y_size > 0 else 6
            noise = torch.randn(num_images, self.noise_size, 1, 1)
            y = torch.tensor(range(num_images))

            if self.on_gpu:
                noise = noise.cuda(self.real_images.device.index)
                y = y.cuda(self.real_images.device.index)

            fake_images = self.generator.forward(noise, y)
            grid = torchvision.utils.make_grid(fake_images, nrow=int(num_images / 2))

            # for tensorboard
            # self.logger.experiment.add_image("example_images", grid, 0)

            # for comet.ml
            self.logger.experiment.log_image(
                grid.detach().cpu().numpy(),
                name="generated images",
                image_channels="first"
            )

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()

        # update generator opt every {self.alternation_interval} steps
        if optimizer_idx == 0 and batch_idx % self.alternation_interval == 0:
            optimizer.step()
            optimizer.zero_grad()

        # update critic opt every step
        if optimizer_idx == 1:
            optimizer.step()
            optimizer.zero_grad()

    def configure_optimizers(self):
        return [
            optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2)),
            optim.Adam(self.critic.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        ]

    def prepare_data(self):
        # self.inception_model = torchvision.models.inception_v3(pretrained=True, progress=True)

        train_resize = transforms.Resize((self.image_width, self.image_height))
        test_resize = transforms.Resize(224, 224)

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

    # def test_dataloader(self):
    #    return DataLoader(self.test_dataset, num_workers=self.dataloader_num_workers, batch_size=self.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        train_group = parser.add_argument_group("Training")
        train_group.add_argument("-mine", "--min-epochs", type=int, default=1, help="Minimum number of epochs to train")
        train_group.add_argument("-maxe", "--max-epochs", type=int, default=1000, help="Maximum number of epochs to train")
        train_group.add_argument("-acb", "--accumulate-grad-batches", type=int, default=1, help="Accumulate gradient batches")
        train_group.add_argument("-dnw", "--dataloader-num-workers", type=int, default=8, help="Number of workers the dataloader uses")

        system_group = parser.add_argument_group("System")
        system_group.add_argument("-ic", "--image-channels", type=int, default=3, help="Generated image shape channels")
        system_group.add_argument("-iw", "--image-width", type=int, default=32, help="Generated image shape width")
        system_group.add_argument("-ih", "--image-height", type=int, default=32, help="Generated image shape height")
        system_group.add_argument("-bs", "--batch-size", type=int, default=32, help="Batch size")
        system_group.add_argument("-lr", "--learning-rate", type=float, default=0.0001, help="Learning rate of both optimizers")
        train_group.add_argument("-b1", "--beta1", type=int, default=0.5, help="Momentum term beta1")
        train_group.add_argument("-b2", "--beta2", type=int, default=0.9, help="Momentum term beta2")
        system_group.add_argument("-z", "--noise-size", type=int, default=100, help="Length of the noise vector")
        system_group.add_argument("-y", "--y-size", type=int, default=10, help="Length of the y/label vector")
        system_group.add_argument("-yes", "--y-embedding-size", type=int, default=10, help="Length of the y/label embedding vector")
        system_group.add_argument("-k", "--alternation-interval", type=int, default=5, help="Amount of steps the critic is trained for each training step of the generator")

        critic_group = parser.add_argument_group("Critic")
        critic_group.add_argument("-clrs", "--critic-leaky-relu-slope", type=float, default=0.2, help="Slope of the leakyReLU activation function in the critic")
        critic_group.add_argument("-cf", "--critic-filters", type=int, default=64, help="Filters in the critic (are multiplied with different powers of 2)")
        critic_group.add_argument("-cl", "--critic-length", type=int, default=2, help="Length of the critic or number of down sampling blocks")
        critic_group.add_argument("-gpt", "--gradient-penalty-term", type=float, default=10, help="Gradient penalty term")

        generator_group = parser.add_argument_group("Generator")
        generator_group.add_argument("-gf", "--generator-filters", type=int, default=64, help="Filters in the generator (are multiplied with different powers of 2)")
        generator_group.add_argument("-gl", "--generator-length", type=int, default=3, help="Length of the generator or number of up sampling blocks (also determines the size of the output image)")

        return parser
