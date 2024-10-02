import numpy as np

class Sigmoidal():
    def __init__(self):
        pass

    def forward(self, x): # x shape deve ser (1, din)
        return 1 / (1 + np.exp(-x))
    
    def backward(self):
        pass

    def __call__(self, x):
        return self.forward(x)