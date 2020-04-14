import math
import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from pytorch_lightning.logging import CometLogger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, ImageNet, LSUN

from ..helpers import inception_score


class GAN(pl.LightningModule):
    def __init__(self, hparams, generator, critic, scorer):
        super().__init__()

        self.hparams = hparams

        if self.hparams.gradient_penalty_coefficient is None:
            if self.hparams.gradient_penalty_strategy == "0-gp":
                self.hparams.gradient_penalty_coefficient = 10
            elif self.hparams.gradient_penalty_strategy == "1-gp":
                self.hparams.gradient_penalty_coefficient = 10
            elif self.hparams.gradient_penalty_strategy == "lp":
                self.hparams.gradient_penalty_coefficient = 0.1
            elif self.hparams.gradient_penalty_strategy == "div":
                self.hparams.gradient_penalty_coefficient = 2
            elif self.hparams.gradient_penalty_strategy == "ct":
                self.hparams.gradient_penalty_coefficient = 10
            elif self.hparams.gradient_penalty_strategy == "none":
                self.hparams.gradient_penalty_coefficient = 0
            else:
                raise ValueError()

        if self.hparams.gradient_penalty_power is None:
            if self.hparams.gradient_penalty_strategy == "0-gp":
                self.hparams.gradient_penalty_power = 2
            elif self.hparams.gradient_penalty_strategy == "1-gp":
                self.hparams.gradient_penalty_power = 2
            elif self.hparams.gradient_penalty_strategy == "lp":
                self.hparams.gradient_penalty_power = 2
            elif self.hparams.gradient_penalty_strategy == "div":
                self.hparams.gradient_penalty_power = 6
            elif self.hparams.gradient_penalty_strategy == "ct":
                self.hparams.gradient_penalty_power = 2
            elif self.hparams.gradient_penalty_strategy == "none":
                self.hparams.gradient_penalty_power = 0
            else:
                raise ValueError()

        if self.hparams.consistency_term_coefficient is None:
            if self.hparams.gradient_penalty_strategy == "0-gp":
                self.hparams.consistency_term_coefficient = 0
            elif self.hparams.gradient_penalty_strategy == "1-gp":
                self.hparams.consistency_term_coefficient = 0
            elif self.hparams.gradient_penalty_strategy == "lp":
                self.hparams.consistency_term_coefficient = 0
            elif self.hparams.gradient_penalty_strategy == "div":
                self.hparams.consistency_term_coefficient = 0
            elif self.hparams.gradient_penalty_strategy == "ct":
                self.hparams.consistency_term_coefficient = 2
            elif self.hparams.gradient_penalty_strategy == "none":
                self.hparams.gradient_penalty_strategy = 0
            else:
                raise ValueError()

        self.generator = generator
        self.critic = critic
        self.scorer = scorer

        self.real_images = None
        self.y = None
        self.experience = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def on_train_start(self):
        if isinstance(self.logger, CometLogger):
            self.logger.experiment.set_model_graph(str(self))
        elif isinstance(self.logger, WandbLogger):
            pass

    def forward(self, x, y):
        output = self.generator(x, y)
        return output

    def critic_loss(self, real_validity, fake_validity):
        if self.hparams.loss_strategy == "wgan":
            real_loss = -real_validity
            fake_loss = fake_validity
        elif self.hparams.loss_strategy == "lsgan":
            real_loss = -(real_validity - 1) ** 2
            fake_loss = fake_validity ** 2
        elif self.hparams.loss_strategy == "hinge":
            real_loss = torch.relu(1.0 - real_validity)
            fake_loss = torch.relu(1.0 + fake_validity)
        elif self.hparams.loss_strategy == "r-hinge":
            relativistic_real_validity = real_validity - fake_validity
            relativistic_fake_validity = fake_validity - real_validity

            real_loss = torch.relu(1.0 - relativistic_real_validity)
            fake_loss = torch.relu(1.0 + relativistic_fake_validity)
        elif self.hparams.loss_strategy == "ra-hinge":
            relativistic_real_validity = real_validity - fake_validity.mean()
            relativistic_fake_validity = fake_validity - real_validity.mean()

            real_loss = torch.relu(1.0 - relativistic_real_validity)
            fake_loss = torch.relu(1.0 + relativistic_fake_validity)
        elif self.hparams.loss_strategy == "ns":
            real_loss = -torch.log(torch.sigmoid(real_validity))
            # noinspection PyTypeChecker
            fake_loss = -torch.log(1.0 - torch.sigmoid(fake_validity))
        else:
            raise ValueError()

        loss = real_loss.mean() + fake_loss.mean()
        return loss.unsqueeze(0)

    def generator_loss(self, real_validity, fake_validity):
        if self.hparams.loss_strategy == "wgan":
            fake_loss = -fake_validity
            loss = fake_loss.mean()
        elif self.hparams.loss_strategy == "lsgan":
            fake_loss = -(fake_validity - 1) ** 2
            loss = fake_loss.mean()
        elif self.hparams.loss_strategy == "hinge":
            fake_loss = -fake_validity
            loss = fake_loss.mean()
        elif self.hparams.loss_strategy == "r-hinge":
            relativistic_real_validity = real_validity - fake_validity
            relativistic_fake_validity = fake_validity - real_validity

            real_loss = torch.relu(1.0 - relativistic_fake_validity)
            fake_loss = torch.relu(1.0 + relativistic_real_validity)

            loss = fake_loss.mean() + real_loss.mean()
        elif self.hparams.loss_strategy == "ra-hinge":
            relativistic_real_validity = real_validity - fake_validity.mean()
            relativistic_fake_validity = fake_validity - real_validity.mean()

            real_loss = torch.relu(1.0 - relativistic_fake_validity)
            fake_loss = torch.relu(1.0 + relativistic_real_validity)

            loss = fake_loss.mean() + real_loss.mean()
        elif self.hparams.loss_strategy == "ns":
            fake_loss = -torch.log(torch.sigmoid(fake_validity))
            loss = fake_loss.mean()
        else:
            raise ValueError()

        return loss.unsqueeze(0)

    def clip_weights(self):
        for weight in self.critic.parameters():
            weight.data.clamp_(-self.hparams.weight_clipping, self.hparams.weight_clipping)

    # TODO: Need to check if gradient penalty works well with multi-scale gradient
    def gradient_penalty(self, real_images, fake_images, y):
        if self.hparams.gradient_penalty_coefficient != 0:
            alpha = torch.rand(real_images.size(0), 1, 1, 1, device=real_images.device)

            if self.hparams.gradient_penalty_strategy == "div":
                # noinspection PyTypeChecker
                interpolates = alpha * fake_images + (1 - alpha) * real_images
            else:
                # noinspection PyTypeChecker
                interpolates = alpha * real_images + (1 - alpha) * fake_images

            interpolates.requires_grad_()

            if self.hparams.multi_scale_gradient:
                scaled_interpolates = self.to_scaled_images(interpolates)
                interpolates_validity = self.critic(interpolates, y, scaled_inputs=scaled_interpolates)
            else:
                interpolates_validity = self.critic(interpolates, y)

            gradients = torch.autograd.grad(
                outputs=interpolates_validity,
                inputs=interpolates,
                grad_outputs=torch.ones_like(interpolates_validity, device=real_images.device),
                create_graph=True
            )[0]

            gradients = gradients.view(gradients.size(0), -1)

            if self.hparams.gradient_penalty_strategy == "0-gp":
                penalties = gradients.norm(dim=1) ** self.hparams.gradient_penalty_power
            elif self.hparams.gradient_penalty_strategy == "1-gp":
                penalties = (gradients.norm(dim=1) - 1) ** self.hparams.gradient_penalty_power
            elif self.hparams.gradient_penalty_strategy == "lp":
                # noinspection PyTypeChecker
                penalties = torch.max(torch.tensor(0.0, device=real_images.device), gradients.norm(dim=1) - 1) ** self.hparams.gradient_penalty_power
            elif self.hparams.gradient_penalty_strategy == "div":
                penalties = gradients.norm(dim=1) ** self.hparams.gradient_penalty_power
            elif self.hparams.gradient_penalty_strategy == "ct":
                penalties = (gradients.norm(dim=1) - 1) ** self.hparams.gradient_penalty_power
            else:
                raise ValueError()

            return self.hparams.gradient_penalty_coefficient * penalties.mean().unsqueeze(0)
        else:
            return 0

    def consistency_term(self, real_images, y, scaled_real_images=None, m=0):
        if self.hparams.consistency_term_coefficient != 0:
            # TODO: Need to check if correct
            d_x1, d_x1_ = self.critic.forward(real_images, y, dropout=0.5, intermediate_output=True, scaled_inputs=scaled_real_images)
            d_x2, d_x2_ = self.critic.forward(real_images, y, dropout=0.5, intermediate_output=True, scaled_inputs=scaled_real_images)

            consistency_term = torch.relu(torch.dist(d_x1, d_x2) + 0.1 * torch.dist(d_x1_, d_x2_) - m)

            return self.hparams.consistency_term_coefficient * consistency_term.unsqueeze(0)
        else:
            return 0

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.critic.train(optimizer_idx == 0)
        self.generator.train(optimizer_idx == 1)

        if optimizer_idx == 0:  # Train critic
            return self.training_step_critic(batch)

        if optimizer_idx == 1:  # Train generator
            return self.training_step_generator(batch)

    def training_step_critic(self, batch):
        self.real_images, self.y = batch

        noise = torch.randn(self.real_images.size(0), self.hparams.noise_size, device=self.real_images.device)

        # if self.experience is not None and self.experience.size(0) == self.real_images.size(0):
        #    fake_images = self.experience.detach()
        #    self.experience = None
        # else:

        if self.hparams.multi_scale_gradient:
            scaled_real_images = self.to_scaled_images(self.real_images)
            fake_images, scaled_fake_images = self.forward(noise, self.y)
            scaled_fake_images = [x.detach() for x in scaled_fake_images]
            fake_images = fake_images.detach()

            real_validity = self.critic(self.real_images, self.y, scaled_inputs=scaled_real_images)
            fake_validity = self.critic(fake_images, self.y, scaled_inputs=scaled_fake_images)

            # TODO: Need to check if gradient penalty works well
            gradient_penalty = self.gradient_penalty(self.real_images, fake_images, self.y)
            consistency_term = self.consistency_term(self.real_images, self.y, scaled_real_images)
        else:
            fake_images = self.forward(noise, self.y).detach()

            real_validity = self.critic(self.real_images, self.y)
            fake_validity = self.critic(fake_images, self.y)

            gradient_penalty = self.gradient_penalty(self.real_images, fake_images, self.y)
            consistency_term = self.consistency_term(self.real_images, self.y)

        loss = self.critic_loss(real_validity, fake_validity)

        if len(self.trainer.lr_schedulers) >= 1:
            critic_lr = self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0]
        else:
            critic_lr = self.hparams.critic_learning_rate

        logs = {"critic_loss": loss, "gradient_penalty": gradient_penalty, "consistency_term": consistency_term, "critic_lr": critic_lr}
        return OrderedDict({"loss": loss + gradient_penalty, "log": logs, "progress_bar": logs})

    def training_step_generator(self, batch):
        self.real_images, self.y = batch

        noise = torch.randn(self.real_images.size(0), self.hparams.noise_size, device=self.real_images.device)

        if self.hparams.multi_scale_gradient:
            scaled_real_images = self.to_scaled_images(self.real_images)
            fake_images, scaled_fake_images = self.forward(noise, self.y)

            real_validity = self.critic(self.real_images, self.y, scaled_inputs=scaled_real_images)
            fake_validity = self.critic(fake_images, self.y, scaled_inputs=scaled_fake_images)
        else:
            fake_images = self.forward(noise, self.y)

            real_validity = self.critic(self.real_images, self.y)
            fake_validity = self.critic(fake_images, self.y)

        # if self.hparams.enable_experience_replay:
        #    rand_image = fake_images[random.randint(0, fake_images.size(0) - 1)].unsqueeze(0)

        #    if self.experience is None:
        #        self.experience = rand_image
        #    else:
        #        self.experience = torch.cat([self.experience, rand_image], dim=0)

        loss = self.generator_loss(real_validity, fake_validity)

        if len(self.trainer.lr_schedulers) >= 2:
            generator_lr = self.trainer.lr_schedulers[1]["scheduler"].get_lr()[0]
        else:
            generator_lr = self.hparams.generator_learning_rate

        logs = {"generator_loss": loss, "generator_lr": generator_lr}
        return OrderedDict({"loss": loss, "log": logs, "progress_bar": logs})

    def to_scaled_images(self, source_images):
        return [
            F.interpolate(source_images, size=2 ** target_size)
            for target_size in range(2, int(math.log2(self.hparams.image_size)))
        ]

    # Logs an image for each class defined as noise size
    def on_epoch_end(self):
        if self.logger:
            if self.hparams.score_iterations > 0:
                outputs = []
                for _ in range(self.hparams.score_iterations):
                    noise = torch.randn(self.hparams.batch_size, self.hparams.noise_size, device=self.real_images.device)
                    y = torch.randint(0, 9, (self.hparams.batch_size,), device=self.real_images.device)

                    if self.hparams.multi_scale_gradient:
                        fake_images = self.forward(noise, y)[0].detach()
                    else:
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
                grid_size = self.hparams.y_size if self.hparams.y_size > 1 else 5

                noise = torch.randn(grid_size ** 2, self.hparams.noise_size, device=self.real_images.device)
                y = torch.tensor(range(grid_size), device=self.real_images.device).repeat(grid_size)

                if self.hparams.multi_scale_gradient:
                    fake_images = self.forward(noise, y)[0].detach()
                else:
                    fake_images = self.forward(noise, y).detach()

                grid = torchvision.utils.make_grid(fake_images, nrow=grid_size, padding=0)

                self.logger.experiment.add_image("example_images", grid, 0)
                self.logger.log_metrics({"ic_score_mean": ic_score_mean.item()})
            elif isinstance(self.logger, WandbLogger):
                grid_size = self.hparams.y_size if self.hparams.y_size > 1 else 1

                noise = torch.randn(grid_size, self.hparams.noise_size, device=self.real_images.device)
                y = torch.tensor(range(grid_size), device=self.real_images.device)

                if self.hparams.multi_scale_gradient:
                    fake_images = self.forward(noise, y)[0].detach()
                else:
                    fake_images = self.forward(noise, y).detach()

                self.logger.log_metrics({"ic_score_mean": ic_score_mean.item()})
                self.logger.experiment.log({
                    "generated_images": [wandb.Image(fake_image, caption=str(idx)) for idx, fake_image in enumerate(fake_images)]
                })
            elif isinstance(self.logger, CometLogger):
                grid_size = self.hparams.y_size if self.hparams.y_size > 1 else 5

                noise = torch.randn(grid_size ** 2, self.hparams.noise_size, device=self.real_images.device)
                y = torch.tensor(range(grid_size), device=self.real_images.device).repeat(grid_size)

                if self.hparams.multi_scale_gradient:
                    fake_images = self.forward(noise, y)[0].detach()
                else:
                    fake_images = self.forward(noise, y).detach()

                grid = torchvision.utils.make_grid(fake_images, nrow=grid_size, padding=0)

                self.logger.experiment.log_image(
                    grid.detach().cpu().numpy(),
                    name="generated_images",
                    image_channels="first"
                )
                self.logger.log_metrics({"ic_score_mean": ic_score_mean.item()})

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # update critic opt every step
        if optimizer_idx == 0:  optimizer.step()
        # update generator opt every {self.alternation_interval} steps
        if optimizer_idx == 1 and batch_idx % self.hparams.alternation_interval == 0: optimizer.step()

        optimizer.zero_grad()

    def configure_optimizers(self):
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hparams.critic_learning_rate, betas=(self.hparams.critic_beta1, self.hparams.critic_beta2))
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.hparams.generator_learning_rate, betas=(self.hparams.generator_beta1, self.hparams.generator_beta2))

        # critic_lr_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=200, gamma=0.1)
        # generator_lr_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=200, gamma=0.1)

        # , [critic_lr_scheduler, generator_lr_scheduler]

        return [critic_optimizer, generator_optimizer]

    def prepare_data(self):
        train_resize = transforms.Resize((self.hparams.image_size, self.hparams.image_size))
        # test_resize = transforms.Resize(224, 224)

        if self.hparams.image_channels == 3:
            train_normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        else:
            train_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # prepare images for the usage with torchvision models: https://pytorch.org/docs/stable/torchvision/models.html
        # test_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([train_resize, transforms.ToTensor(), train_normalize])
        # test_transform = transforms.Compose([test_resize, transforms.ToTensor(), test_normalize])

        if self.hparams.dataset == "mnist":
            self.train_dataset = MNIST(self.hparams.dataset_path, train=True, download=True, transform=train_transform)
            # self.test_dataset = MNIST(self.hparams.dataset_path, train=False, download=True, transform=test_transform)
        elif self.hparams.dataset == "fashion_mnist":
            self.train_dataset = FashionMNIST(self.hparams.dataset_path, train=True, download=True, transform=train_transform)
            # self.test_dataset = FashionMNIST(self.hparams.dataset_path, train=False, download=True, transform=test_transform)
        elif self.hparams.dataset == "cifar10":
            self.train_dataset = CIFAR10(self.hparams.dataset_path, train=True, download=True, transform=train_transform)
            # self.test_dataset = CIFAR10(self.hparams.dataset_path, train=False, download=True, transform=test_transform)
        elif self.hparams.dataset == "image_net":
            self.train_dataset = ImageNet(self.hparams.dataset_path, train=True, download=True, transform=train_transform)
            # self.test_dataset = ImageNet(self.hparams.dataset_path, train=False, download=True, transform=test_transform)
        elif self.hparams.dataset == "lsun":
            self.train_dataset = LSUN(self.hparams.dataset_path + "/lsun", classes=[cls + "_train" for cls in self.hparams.dataset_classes], transform=train_transform)
            # self.test_dataset = LSUN(self.hparams.dataset_path, classes=[cls + "_test" for cls in self.hparams.dataset_classes], transform=test_transform)
        else:
            raise NotImplementedError("Custom dataset is not implemented yet")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.hparams.dataloader_num_workers, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("-mine", "--min-epochs", type=int, default=1, help="Minimum number of epochs to train")
        parser.add_argument("-maxe", "--max-epochs", type=int, default=1000, help="Maximum number of epochs to train")
        parser.add_argument("-agb", "--accumulate-grad-batches", type=int, default=1, help="Number of gradient batches to accumulate")
        parser.add_argument("-dnw", "--dataloader-num-workers", type=int, default=4, help="Number of workers the dataloader uses")
        parser.add_argument("-cb1", "--critic-beta1", type=float, default=0.0, help="Momentum term beta1 of the critic optimizer")
        parser.add_argument("-cb2", "--critic-beta2", type=float, default=0.9, help="Momentum term beta2 of the critic optimizer")
        parser.add_argument("-gb1", "--generator-beta1", type=float, default=0.0, help="Momentum term beta1 of the generator optimizer")
        parser.add_argument("-gb2", "--generator-beta2", type=float, default=0.9, help="Momentum term beta2 of the generator optimizer")
        parser.add_argument("-v", "--score-iterations", type=int, default=50, help="Number of score iterations each epoch")
        parser.add_argument("-msg", "--multi-scale-gradient", action="store_true", help="Enable Multi-Scale Gradient")
        parser.add_argument("-msgc", "--multi-scale-gradient-combiner", type=str, choices=["simple", "lin_cat", "cat_lin"], default="cat_lin")
        parser.add_argument("-wi", "--weight-init", type=str, choices=["he", "dcgan", "default"], default="he")

        parser.add_argument("-ic", "--image-channels", type=int, default=3, help="Generated image shape channels")
        parser.add_argument("-is", "--image-size", type=int, default=128, help="Generated image size")
        parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size")

        # TTUR: https://arxiv.org/abs/1706.08500
        parser.add_argument("-clr", "--critic-learning-rate", type=float, default=1e-4, help="Learning rate of the critic optimizers")
        parser.add_argument("-glr", "--generator-learning-rate", type=float, default=1e-4, help="Learning rate of the generator optimizers")

        parser.add_argument("-ls", "--loss-strategy", type=str, choices=["lsgan", "wgan", "mm", "hinge", "ns", "r-hinge", "ra-hinge"], default="ra-hinge")
        parser.add_argument("-gs", "--gradient-penalty-strategy", type=str, choices=[
            "1-gp",  # Original 2-sided WGAN-GP
            "0-gp",  # Improving Generalization and Stability of Generative Adversarial Networks: https://openreview.net/forum?id=ByxPYjC5KQ
            "lp",  # 1-Sided: On the regularization of Wasserstein GANs: https://arxiv.org/abs/1709.08894
            "div",
            "ct",
            "none"
        ], default="1-gp")

        parser.add_argument("-z", "--noise-size", type=int, default=100, help="Length of the noise vector")
        parser.add_argument("-y", "--y-size", type=int, default=10, help="Length of the y/label vector")
        parser.add_argument("-yes", "--y-embedding-size", type=int, default=10, help="Length of the y/label embedding vector")
        parser.add_argument("-k", "--alternation-interval", type=int, default=1, help="Amount of steps the critic is trained for each training step of the generator")
        parser.add_argument("-gpc", "--gradient-penalty-coefficient", type=float, default=None, help="Gradient penalty coefficient")
        parser.add_argument("-gpp", "--gradient-penalty-power", type=float, default=None, help="Gradient penalty coefficient")
        parser.add_argument("-ctw", "--consistency-term-coefficient", type=float, default=0, help="Consistency term coefficient")
        parser.add_argument("-wc", "--weight-clipping", type=float, default=0.01, help="Weights of the critic gets clipped at this point")

        parser.add_argument("-gf", "--generator-filters", type=int, default=64, help="Number of filters in the generator")
        parser.add_argument("-cf", "--critic-filters", type=int, default=64, help="Number of filters in the critic")
        parser.add_argument("-eer", "--enable-experience-replay", action="store_true", help="Find paper for this")

        parser.add_argument("--dataset", type=str, choices=["custom", "cifar10", "mnist", "fashion_mnist", "lsun", "image_net"], required=True)
        parser.add_argument("--dataset-path", type=str, default=os.getcwd() + "/.datasets")
        parser.add_argument("--dataset-classes", type=int, nargs="+", default=["church_outdoor"])

        return parser
