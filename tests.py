import torch
import random
import numpy as np

from modules.activation_functions import Sigmoidal
from modules.structures import Layer, Network

def test_forward():
    layer = Layer(6, 5)
    x = [random.random() for _ in range(6)]
    data = layer(x)

    layer_torch = torch.nn.Linear(6, 5)
    layer_torch.weight.data = torch.tensor(layer.W)
    layer_torch.bias.data = torch.tensor(layer.b)
    data_torch = layer_torch(torch.tensor(x)).tolist()

    error = ((np.array(data) - np.array(data_torch)) ** 2).mean()

    if error < 1e-8:
        return 0
    else:
        return -1

assert test_forward() == 0