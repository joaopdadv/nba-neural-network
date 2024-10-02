import numpy as np

class MSELoss():
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return ((np.array(pred) - np.array(true)) ** 2).mean()
    
    def __call__(self, pred, true):
        return self.forward(pred, true)
    
    def backward(self):
        n = len(self.pred)  # Número de elementos
        return (2 * (np.array(self.pred) - np.array(self.true))) / n  # Gradiente da MSE