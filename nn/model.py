# Simple model that chains all the layers, and returns iterable params

import numpy as np



class Model:
    def __init__(self, layers):
        self.layers = layers
    

    def forward(self, x):
        #Pass input through all layers
        for layer in self.layers:
            x = layer.forward(x)
        return x
    

    def backward(self, dout):
        #Pass the gradient back through all layers
        for layer in self.layers:
            dout = layer.backward(dout)
        return dout
    
    def params(self):
        #Collects all learnable params
        pairs = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                pairs.extend(layer.params())
        return pairs
    def zero_grad(self):
        for _, g in self.params():
            g[...]  = 0.0