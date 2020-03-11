import torch


# https://machinelearningmastery.com/divergence-between-probability-distributions/
def kl_divergence(p, q):
    return torch.sum(p * torch.log2(p / q), dim=1)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
