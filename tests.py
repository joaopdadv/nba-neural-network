import torch
import random
import numpy as np

from modules.activation_functions import Sigmoidal, Softmax
from modules.structures import Layer, Network
from modules.loss_functions import MSELoss

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

def test_sigmoidal():
    sigmoidal = Sigmoidal()
    layer = Layer(6, 5)
    x = [random.random() for _ in range(6)]
    data = sigmoidal(layer(x))

    sigmoidal_torch = torch.nn.Sigmoid()
    layer_torch = torch.nn.Linear(6, 5)
    layer_torch.weight.data = torch.tensor(layer.W)
    layer_torch.bias.data = torch.tensor(layer.b)
    data_torch = sigmoidal_torch(layer_torch(torch.tensor(x))).tolist()

    error = ((np.array(data) - np.array(data_torch)) ** 2).mean()

    if error < 1e-8:
        return 0
    else:
        return -1
    
def test_softmax():
    layer1 = Layer(6, 10)
    sigmoidal1 = Sigmoidal()
    layer2 = Layer(10, 2)
    s = Softmax()
    network = Network([layer1, sigmoidal1, layer2, s])

    x = [random.random() for _ in range(6)]
    data = network(x)

    layer1_torch = torch.nn.Linear(6, 10)
    layer1_torch.weight.data = torch.tensor(layer1.W)
    layer1_torch.bias.data = torch.tensor(layer1.b)
    sigmoidal_torch = torch.nn.Sigmoid()
    layer2_torch = torch.nn.Linear(10, 2)
    layer2_torch.weight.data = torch.tensor(layer2.W)
    layer2_torch.bias.data = torch.tensor(layer2.b)
    s_torch = torch.nn.Softmax(dim=0)
    network_torch = torch.nn.Sequential(layer1_torch, sigmoidal_torch, layer2_torch, s_torch)

    data_torch = network_torch(torch.tensor(x)).tolist()

    error = ((np.array(data) - np.array(data_torch)) ** 2).mean()

    if error < 1e-8:
        return 0
    else:
        return -1
    
def test_mse_loss():
    
    layer1 = Layer(6, 10)
    sigmoidal1 = Sigmoidal()
    layer2 = Layer(10, 2)
    s = Softmax()
    network = Network([layer1, sigmoidal1, layer2, s])

    x = [random.random() for _ in range(6)]
    prediction = network.forward(x)  # tensor com 2 elementos

    target = [random.random() for _ in range(2)]  # tensor com 2 elementos

    mse = MSELoss()
    loss = mse.forward(prediction, target)

    loss_torch = torch.nn.MSELoss()
    loss_torch = loss_torch(torch.tensor(prediction), torch.tensor(target)).tolist()

    error = ((np.array(loss) - np.array(loss_torch)) ** 2).mean()

    if error < 1e-8:
        return 0
    else:
        return -1

def test_mse_grad():
    # Inicialização da rede
    layer1 = Layer(6, 10)
    sigmoidal1 = Sigmoidal()
    layer2 = Layer(10, 2)
    s = Softmax()
    network = Network([layer1, sigmoidal1, layer2, s])

    # Gerar entrada e saída
    x = [random.random() for _ in range(6)]
    prediction = network.forward(x)  # tensor com 2 elementos

    target = [random.random() for _ in range(2)]  # tensor com 2 elementos

    # Cálculo da perda MSE na implementação customizada
    mse = MSELoss()
    mse.forward(prediction, target)

    # Cálculo da perda MSE usando PyTorch
    prediction_tensor = torch.tensor(prediction, requires_grad=True)  # Habilitar gradientes
    target_tensor = torch.tensor(target)
    
    loss_torch = torch.nn.MSELoss()(prediction_tensor, target_tensor)
    loss_torch.backward()  # Calcula os gradientes

    # Obter gradientes do PyTorch
    grad_torch = prediction_tensor.grad.numpy()

    # Cálculo do gradiente na implementação customizada
    grad_custom = mse.backward()

    print(grad_torch)
    print(grad_custom)

    # Comparar os gradientes
    error = ((np.array(grad_custom) - np.array(grad_torch)) ** 2).mean()

    if error < 1e-8:
        return 0
    else:
        return -1

assert test_mse_grad() == 0
assert test_mse_loss() == 0
assert test_softmax() == 0
assert test_sigmoidal() == 0
assert test_forward() == 0