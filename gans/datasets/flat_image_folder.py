import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import is_image_file


class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root + "/"
        self.transform = transform
        self.loader = loader

        self.files = sorted([file for file in os.listdir(self.root) if is_image_file(file)])

    def __getitem__(self, index: int):
        path = self.root + self.files[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 0

    def __len__(self):
        return len(self.files)
