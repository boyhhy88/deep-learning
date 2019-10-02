import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    
    return A

def relu(Z):
    A = np.maximum(0, Z)
    
    return A

def relu_backward(dA, Z):
    assert(dA.shape == Z.shape)
    # numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0): create an array.
    dZ = np.array(dA, copy = True)
    # numpy boolean indexing
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, Z):
    assert(dA.shape == Z.shape)
    Y = sigmoid(Z)
    dZ = dA * Y * (1 - Y)
    
    return dZ

