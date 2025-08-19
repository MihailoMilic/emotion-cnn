import numpy as np
from nn.layers import Conv2D, ReLU, Linear, Flatten, MaxPool2D
from .model import Model
from .train import train
from .softmax import softmax_loss
from .SGD import SGD

num_classes = 7


model = Model([
    Conv2D(C_in=1, C_out=8, kH=3, kW= 3, stride =1, pad =1),
    ReLU(),
    MaxPool2D(pool_h=2, pool_w= 2, stride=2),
    Conv2D(C_in=8, C_out=16, kH=3, kW=3, stride=1, pad=1),
    ReLU(),
    MaxPool2D(pool_h=2, pool_w=2 , stride=2),
    Flatten(),
    Linear(16*12*12, 64),
    ReLU(),
    Linear(64, num_classes)
    ], 
    )


loss_fn = softmax_loss
optim = SGD(lr = 5e-3, weight_decay= 1e-4)

train_npz = np.load('data/train48.npz')
test_npz = np.load('data/test48.npz')

X_train = train_npz['X'].astype(np.float32)
y_train = train_npz['y'].astype(np.int64)

X_val = test_npz['X'].astype(np.float32)
y_val = train_npz['y'].astype(np.int64)

train(model, loss_fn, optim, X_train, y_train, X_val, y_val, batch_size=64, epochs=10)

