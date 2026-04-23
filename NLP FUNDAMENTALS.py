import numpy as np 

def relu(x):
    return np.maximum(0, x)

def forward_pass(x, params):
    W1 , b1 , W2 , b2 = params
    z1 = x@ W1 + b1
    h1 = relu(z1)
    z2 = h1 @ W2 + b2
    
    return z1 , h1 , z2

x_batch = np.array([[0.5, -1.0, 2.0], [1.5, 0.2, -0.3]])
params = (
    np.array([[0.2, -0.4], [0.7, 0.5], [-0.3, 0.8]]),  # W1
    np.array([0.1, -0.2]),                             # b1
    np.array([[0.6, -0.1], [-0.5, 0.9]]),              # W2
    np.array([0.0, 0.05]))                             # b2

z1, h1, z2 = forward_pass(x_batch, params)
print(z1 , h1 , z2) 









