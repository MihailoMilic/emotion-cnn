import numpy as np

class SGD:
    #Simple Stocastic Gradient descent for now
    def __init__(self, lr = 1E-5, weight_decay = 0.0):
        self.wd = float(weight_decay)
        self.lr = lr
    def step(self, model):
        for param, grad in model.params():
            if self.wd != 0.0:
                grad = grad +self.wd * param
            param -= self.lr * grad
    

