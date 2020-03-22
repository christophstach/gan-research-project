import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.logging import CometLogger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from gans.helpers.metrics import inception_score
import wandb

class WGANGP(pl.LightningModule):
    def __init__(self, hparams, generator, critic, scorer):
        super().__init__()

        self.hparams = hparams

        self.generator = generator
        self.critic = critic
        self.scorer = scorer

        self.real_images = None
        self.y = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def on_train_start(self):
        if isinstance(self.logger, CometLogger):
            self.logger.experiment.set_model_graph(str(self))
        elif isinstance(self.logger, WandbLogger):
            pass

        if self.hparams.pretrain_enabled:
            train_set, val_set = random_split(self.train_dataset, [int(len(self.train_dataset) * 0.8), int(len(self.train_dataset) * 0.2)])

            train_loader = DataLoader(train_set, num_workers=self.hparams.dataloader_num_workers, batch_size=self.hparams.batch_size)
            val_loader = DataLoader(val_set, num_workers=self.hparams.dataloader_num_workers, batch_size=self.hparams.batch_size)

            critic_trainer = Trainer(
                min_epochs=self.hparams.pretrain_min_epochs,
                max_epochs=self.hparams.pretrain_max_epochs,
                gpus=self.hparams.gpus,
                nb_gpu_nodes=self.hparams.nodes,
                accumulate_grad_batches=self.hparams.pretrain_accumulate_grad_batches,
                progress_bar_refresh_rate=20,
                early_stop_callback=False,
                checkpoint_callback=False,
                logger=False,
                distributed_backend="dp"
            )

            self.critic.pretrain = True
            critic_trainer.fit(self.critic, train_loader, val_loader)
            self.critic.pretrain = False

    def forward(self, x, y):
        return self.generator(x, y)

    def critic_loss(self, real_validity, fake_validity):
        if self.hparams.loss_type in ["wgan-gp1", "wgan-gp2", "wgan-gp-div"]:
            return (fake_validity.mean() - real_validity.mean()).unsqueeze(0)
        elif self.hparams.loss_type == "lsgan":
            return (0.5 * ((real_validity - 1) ** 2).mean() + 0.5 * (fake_validity ** 2).mean()).unsqueeze(0)

    def generator_loss(self, fake_validity):
        if self.hparams.loss_type in ["wgan-gp1", "wgan-gp2", "wgan-gp-div"]:
            return (-fake_validity.mean()).unsqueeze(0)
        elif self.hparams.loss_type == "lsgan":
            return (0.5 * ((fake_validity - 1) ** 2).mean()).unsqueeze(0)

    def clip_weights(self):
        for weight in self.critic.parameters():
            weight.data.clamp_(-self.hparams.weight_clipping, self.hparams.weight_clipping)

    def gradient_penalty(self, real_images, fake_images, y):
        """Calculates the gradient penalty loss for WGAN GP"""

        if self.hparams.loss_type == "wgan-gp1":
            # Random weight term for interpolation between real and fake samples
            alpha = torch.rand(real_images.size(0), 1, 1, 1, device=real_images.device)
            # Get random interpolation between real and fake samples
            interpolates = alpha * real_images + ((1 - alpha) * fake_images)
        elif self.hparams.loss_type == "wgan-gp2":
            # Random weight term for interpolation between real and fake samples
            alpha = torch.randn_like(real_images, device=real_images.device)
            # Get random interpolation between real and fake samples
            interpolates = real_images + alpha * (fake_images - real_images)
        else:
            raise NotImplementedError()

        interpolates.requires_grad_()

        # Get gradient w.r.t. interpolates
        interpolate_validity = self.critic(interpolates, y)
        gradients = torch.autograd.grad(
            outputs=interpolate_validity, inputs=interpolates, grad_outputs=torch.ones_like(interpolate_validity), create_graph=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        penalty = (((gradients.norm(2, dim=1) - 1) ** 2).mean()).unsqueeze(0)

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

        noise = torch.randn(self.real_images.size(0), self.hparams.noise_size, device=self.real_images.device)

        fake_images = self.forward(noise, self.y).detach()
        real_validity = self.critic(self.real_images, self.y)
        fake_validity = self.critic(fake_images, self.y)

        if self.hparams.loss_type in ["wgan-gp1", "wgan-gp2"]:
            gradient_penalty = self.hparams.gradient_penalty_term * self.gradient_penalty(self.real_images, fake_images, self.y)
        elif self.hparams.loss_type == "wgan-gp-div":
            gradient_penalty = self.divergence_gradient_penalty(real_validity, fake_validity, self.real_images, fake_images)
        else:
            gradient_penalty = 0

        loss = self.critic_loss(real_validity, fake_validity)
        logs = {"critic_loss": loss, "gradient_penalty": gradient_penalty, "critic_lr": self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0]}
        return OrderedDict({"loss": loss + gradient_penalty, "log": logs, "progress_bar": logs})

    def training_step_generator(self, batch):
        self.real_images, self.y = batch

        noise = torch.randn(self.real_images.size(0), self.hparams.noise_size, device=self.real_images.device)

        fake_images = self.forward(noise, self.y)
        fake_validity = self.critic(fake_images, self.y)
        loss = self.generator_loss(fake_validity)

        logs = {"generator_loss": loss, "generator_lr": self.trainer.lr_schedulers[1]["scheduler"].get_last_lr()[0]}
        return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:  # Train critic
            return self.training_step_critic(batch)

        if optimizer_idx == 1:  # Train generator
            return self.training_step_generator(batch)

    def validation_epoch_end(self, outputs):
        ic_score_mean = torch.stack([x["ic_score"] for x in outputs]).mean()

        logs = {"ic_score_mean": ic_score_mean}
        return OrderedDict({"ic_score_mean": ic_score_mean, "progress_bar": logs, "log": logs})

    # Logs an image for each class defined as noise size
    def on_epoch_end(self):
        if self.logger:
            noise = torch.randn(self.hparams.y_size ** 2, self.hparams.noise_size, device=self.real_images.device)
            y = torch.tensor(range(self.hparams.y_size), device=self.real_images.device).repeat(self.hparams.y_size)

            fake_images = self.forward(noise, y)
            grid = torchvision.utils.make_grid(fake_images, nrow=self.hparams.y_size, padding=0)

            if self.hparams.validations > 0:
                outputs = []
                for _ in range(self.hparams.validations):
                    noise = torch.randn(self.hparams.batch_size, self.hparams.noise_size, device=self.real_images.device)
                    y = torch.randint(0, 9, (self.hparams.batch_size,), device=self.real_images.device)

                    fake_images = self.forward(noise, y).detach()
                    fake_images = F.interpolate(fake_images, (299, 299))

                    if fake_images.size(1) == 1:
                        fake_images = torch.stack([
                            fake_images,
                            fake_images,
                            fake_images
                        ], dim=1).squeeze()

                    logits = F.softmax(self.scorer(fake_images), dim=1)
                    ic_score = inception_score(logits)
                    outputs.append({"ic_score": ic_score})
                ic_score_mean = torch.stack([x["ic_score"] for x in outputs]).mean()
            else:
                ic_score_mean = torch.tensor(0, device=self.real_images.device)

            if isinstance(self.logger, TensorBoardLogger):
                # for tensorboard
                self.logger.experiment.add_image("example_images", grid, 0)
                self.logger.log_metrics({"ic_score_mean": ic_score_mean.item()})
            elif isinstance(self.logger, WandbLogger):
                self.logger.log_metrics({"ic_score_mean": ic_score_mean.item()})
                self.logger.experiment.log({
                    "generated_images": [
                        wandb.Image(grid.detach().cpu().numpy(), caption="image_grid")
                    ]
                })
            elif isinstance(self.logger, CometLogger):
                # for comet.ml and wandb
                self.logger.experiment.log_image(
                    grid.detach().cpu().numpy(),
                    name="generated_images",
                    image_channels="first"
                )
                self.logger.log_metrics({"ic_score_mean": ic_score_mean.item()})

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # update critic opt every step
        if optimizer_idx == 0:
            if self.hparams.warmup_enabled:
                for param in self.critic.features.parameters():
                    param.requires_grad = self.trainer.current_epoch >= self.hparams.warmup_epochs

            optimizer.step()

            if self.hparams.loss_type == "wgan-wc":
                self.clip_weights()

            optimizer.zero_grad()

        # update generator opt every {self.alternation_interval} steps
        if optimizer_idx == 1 and batch_idx % self.hparams.alternation_interval == 0:
            optimizer.step()
            optimizer.zero_grad()

    def configure_optimizers(self):
        if self.hparams.loss_type in ["wgan-gp1", "wgan-gp2", "wgan-gp-div", "lsgan"]:
            critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta1, self.hparams.beta2))
            generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta1, self.hparams.beta2))
        elif self.hparams.loss_type == "wgan-wc":
            critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.learning_rate)
            generator_optimizer = optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError()

        critic_lr_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=200, gamma=0.1)
        generator_lr_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=200, gamma=0.1)

        return [critic_optimizer, generator_optimizer], [critic_lr_scheduler, generator_lr_scheduler]

    def prepare_data(self):
        train_resize = transforms.Resize((self.hparams.image_size, self.hparams.image_size))
        test_resize = transforms.Resize(224, 224)

        if self.hparams.image_channels == 3:
            train_normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        else:
            train_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # prepare images for the usage with torchvision models: https://pytorch.org/docs/stable/torchvision/models.html
        test_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([train_resize, transforms.ToTensor(), train_normalize])
        test_transform = transforms.Compose([test_resize, transforms.ToTensor(), test_normalize])

        if self.hparams.dataset == "mnist":
            train_set = MNIST(os.getcwd() + "/.datasets", train=True, download=True, transform=train_transform)
            test_set = MNIST(os.getcwd() + "/.datasets", train=False, download=True, transform=test_transform)
        elif self.hparams.dataset == "fashion_mnist":
            train_set = FashionMNIST(os.getcwd() + "/.datasets", train=True, download=True, transform=train_transform)
            test_set = FashionMNIST(os.getcwd() + "/.datasets", train=False, download=True, transform=test_transform)
        elif self.hparams.dataset == "cifar10":
            train_set = CIFAR10(os.getcwd() + "/.datasets", train=True, download=True, transform=train_transform)
            test_set = CIFAR10(os.getcwd() + "/.datasets", train=False, download=True, transform=test_transform)
        else:
            raise NotImplementedError("Custom dataset is not implemented yet")

        self.train_dataset = train_set
        # self.test_dataset, self.val_dataset = random_split(test_set, [len(test_set) - 1000, 1000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.hparams.dataloader_num_workers, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        train_group = parser.add_argument_group("Training")
        train_group.add_argument("-mine", "--min-epochs", type=int, default=1, help="Minimum number of epochs to train")
        train_group.add_argument("-maxe", "--max-epochs", type=int, default=1000, help="Maximum number of epochs to train")
        train_group.add_argument("-agb", "--accumulate-grad-batches", type=int, default=1, help="Number of gradient batches to accumulate")
        train_group.add_argument("-dnw", "--dataloader-num-workers", type=int, default=4, help="Number of workers the dataloader uses")
        train_group.add_argument("-b1", "--beta1", type=float, default=0.5, help="Momentum term beta1")
        train_group.add_argument("-b2", "--beta2", type=float, default=0.999, help="Momentum term beta2")
        train_group.add_argument("-v", "--validations", type=int, default=20, help="Number of validations each epoch")

        system_group = parser.add_argument_group("System")
        system_group.add_argument("-ic", "--image-channels", type=int, default=3, help="Generated image shape channels")
        system_group.add_argument("-iw", "--image-size", type=int, default=32, help="Generated image size")
        system_group.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size")
        system_group.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate of both optimizers")
        system_group.add_argument("-lt", "--loss-type", type=str, choices=["wgan-gp1", "wgan-gp2", "wgan-wc", "lsgan", "wgan-gp-div"], default="wgan-gp1")

        system_group.add_argument("-we", "--warmup-enabled", type=bool, default=False, help="Enables freezing of feature layers in the beginning of the training")
        system_group.add_argument("-wi", "--warmup-epochs", type=int, default=5, help="Number of epochs to freeze the critics feature parameters")

        system_group.add_argument("-z", "--noise-size", type=int, default=100, help="Length of the noise vector")
        system_group.add_argument("-y", "--y-size", type=int, default=10, help="Length of the y/label vector")
        system_group.add_argument("-yes", "--y-embedding-size", type=int, default=10, help="Length of the y/label embedding vector")
        system_group.add_argument("-k", "--alternation-interval", type=int, default=5, help="Amount of steps the critic is trained for each training step of the generator")

        critic_group = parser.add_argument_group("Critic")
        critic_group.add_argument("-gpt", "--gradient-penalty-term", type=float, default=10, help="Gradient penalty term")
        critic_group.add_argument("-wc", "--weight-clipping", type=float, default=0.01, help="Weights of the critic gets clipped at this point")

        pretrain_group = parser.add_argument_group("Pretrain")
        pretrain_group.add_argument("-pe", "--pretrain-enabled", type=bool, default=False, help="Enables pretraining of the critic with an classification layer on the real data")
        pretrain_group.add_argument("-pmine", "--pretrain-min-epochs", type=int, default=1, help="Minimum pretrain epochs")
        pretrain_group.add_argument("-pmaxe", "--pretrain-max-epochs", type=int, default=50, help="Maximum pretrain epochs")
        pretrain_group.add_argument("-pagb", "--pretrain-accumulate-grad-batches", type=float, default=1, help="Number of gradient batches to accumulate during pretraining")

        generator_group = parser.add_argument_group("Generator")

        return parser
