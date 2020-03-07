import torch


def to_one_hot(tensor, num_classes):
    y = torch.eye(num_classes)
    return y[tensor]
