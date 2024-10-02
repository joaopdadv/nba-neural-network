import numpy as np

class MSELoss():
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return ((pred - true) ** 2).mean()
    
    def __call__(self, pred, true):
        return self.forward(pred, true)
    
    def backward(self):
        pass