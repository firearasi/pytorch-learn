import torch
from torch import nn
from torch.autograd import Variable

import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST

import numpy as np

import matplotlib.pyplot as plt

import torch.nn.functional as F


# R
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.FloatTensor(np.random.normal(mu, sigma, (1, n)))


# I
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)


# G
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.map1(x)
        out = F.elu(out)
        out = self.map2(out)
        out = F.sigmoid(out)
        out = self.map3(out)


# D
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.elu(self.map1(x))
        out = F.elu(self.map2(out))
        out = F.sigmoid(self.map3(out))








