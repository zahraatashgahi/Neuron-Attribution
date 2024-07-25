import torch
import numpy as np
import torch.nn as nn
import functorch
from functorch import jacrev
from functorch import vmap
from torch.autograd.functional import jacobian


from functorch import jacrev
from functorch import vmap
def MSE(output, label):
    return nn.MSELoss()(output.squeeze(), label)


class parameter_schedule:
    def __init__(self, lambda_1, lambda_2, epoch):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.switch_epoch = epoch

    def get_reg(self, epoch):
        if epoch < self.switch_epoch:
            return 0, 0
        else:
            return self.lambda_1, self.lambda_2


def cosine_similarity(w1, w2):
    return torch.dot(w1, w2).abs() / (torch.norm(w1, 2) * torch.norm(w2, 2))


def weight_correlation(weights, device='cpu'):
    h_dim = weights.shape[0]

    weight_corr = torch.tensor(0., requires_grad=True).to(device)
    weights = weights.clone().requires_grad_(True)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    for neuron_i in range(1, h_dim):
        for neuron_j in range(0, neuron_i):
            pairwise_corr = cosine_similarity(weights[neuron_i, :], weights[neuron_j, :])
            weight_corr = weight_corr + pairwise_corr.norm(p=1)

    return weight_corr / (h_dim * (h_dim - 1) / 2)


def kl_divergence(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD



