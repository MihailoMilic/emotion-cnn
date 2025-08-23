# Simple model that chains all the layers, and returns iterable params

import numpy as np



class Model:
    def __init__(self, layers, debug = False):
        self.layers = layers
        self.debug = debug
    

    def forward(self, x,):
        #Pass input through all layers
        for layer in self.layers:
            x = layer.forward(x)
            # if self.debug is True:
            #     print(f'layer.__class__.__name__ : {x.shape}')
        return x
    

    def backward(self, dout):
        #Pass the gradient back through all layers
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
            if self.debug is True:
                print(f'{layer.__class__.__name__} : {dout.shape}')
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