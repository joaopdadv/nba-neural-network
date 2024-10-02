import numpy as np
import math

class Sigmoidal:
    def __init__(self):
        pass
    
    def forward(self, x): # x shape deve ser (1, din)
        # Aplica a função sigmoidal em cada elemento de x
        return [1 / (1 + math.exp(-xi)) for xi in x] # retorna um vetor de tamanho din

    def backward(self):
        pass

    def __call__(self, x):
        return self.forward(x)
    
class Softmax():
    def __init__(self):
        pass

    def forward(self, x):
        return np.exp(x) / np.exp(x).sum()
    
    def backward(self, gradout):
        pass

    def __call__(self, x):
        return self.forward(x)