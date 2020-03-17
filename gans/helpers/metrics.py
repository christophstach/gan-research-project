import torch


# https://machinelearningmastery.com/divergence-between-probability-distributions/
def kl_divergence(p, q):
    return torch.sum(p * torch.log2(p / q), dim=1)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def inception_score(p_yx, eps=1e-16):
    # calculate p(y)
    p_y = p_yx.mean().unsqueeze(0)
    # kl divergence for each image
    kl_d = p_yx * (torch.log(p_yx + eps) - torch.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(dim=1)
    # average over images
    avg_kl_d = sum_kl_d.mean()
    # undo the logs
    is_score = avg_kl_d.exp()

    return is_score
