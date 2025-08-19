import numpy as np


def softmax_loss(x, y):
    #x : model results (N,C)
    #   N - example number
    #   C - classes of outputs (not channels anymore)
    #y : array of correct predictions. 
    shifted = x - np.max(x, axis =1 , keepdims = True) #this is done to prevent overflow when exponantiated later, probabilities stay the same due to the mathematics behind exponents
    #keepdims gives us (N, 1), and without it wouldve been (N,) which is important because we want to subtract it from X (N, C). This way dimensions align and subtraction can be broadcasted. 
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis =1 , keepdims=True) #using broadcasting (N,C) / (N,1), again thanks to keepdims = True
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N  #indexing probs with y gives us the probability for the correct class in each example, which is the only one we care about for calculating loss.
    
    dx = probs.copy()
    dx[np.arange(N), y] -=1 # ∂L/∂x_i,c = p_i,c - 1{c=y_i}
    dx /= N
    return loss, dx
