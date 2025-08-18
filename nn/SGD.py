import numpy as np

class SGD:
    #Simple Stocastic Gradient descent for now
    def __init__(self, params, lr = 1E-02):
        self.params = params
        self.lr = lr
    def step(self):
        for param, grad in self.params:
            param -= self.lr * grad
    
    def zero_grad(self):
        #Reset gradients for next backpropagation.
        for _, grad in self.params:
            grad.fill(0)

