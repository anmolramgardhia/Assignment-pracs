import numpy as np 

def relu(x):
    return np.maximum(0, x)

def forward_pass(x, params):
    W1 , b1 , W2 , b2 = params
    z1 = x@ W1 + b1
    h1 = relu(z1)
    z2 = h1 @ W2 + b2
    
    return z1 , h1 , z2


