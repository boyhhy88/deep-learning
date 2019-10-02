import math
import numpy as np
import matplotlib.pyplot as plt
from util_func import *


# initialize parameters: weight matrices and bias vectors for each layer

def init_params(layer_dims, init_method, seed = 0):
    """
    Arguments: 
    layer_dims: python array, layer_dims[l] is the number of units in the lth layer. 
                l = 0 is the input layer, the last l is the output layer.
    init_method: choose which initialization method to use: 
                "random_normal":  random values following standard normal distribution.
                "he": He initialization
                "xavier": Xavier initialization
    seed: random seed
    
    Returns:
    params: python dictionary containing weight matrices wl and bias vectors bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
            Wl has the dimension of (layer_dims[l], layer_dims[l - 1]).
            bl has the dimension of (layer_dims[l], 1)
            
    Use random initialization for the weight matrices, and use zeros initialization for the biases.
    """
    
    np.random.seed(seed)
    
    params = {} # parameters to be returned
    
    L = len(layer_dims) # total number of layers, including the input layer. 
    
    for l in range(1, L):
        if init_method == "random_normal":
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 10
        elif init_method == "he":
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        elif init_method == "xavier":
             params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return params      




# initialize velocities for momentum optimization

def init_momentum(layer_dims):
    """
    Arguments: 
    layer_dims: python array, layer_dims[l] is the number of units in the lth layer. 
                l = 0 is the input layer, the last l is the output layer.
    
    Returns:
    v: python dictionary containing velocities for each layer, 
            v['dW1'], v['dW2'], ..., v['dWl'], ..., v['db1'], v['db2'], ..., v['dbl'], ...  
            v['dWl'] has the dimension of (layer_dims[l], layer_dims[l - 1])
            v['dbl'] has the dimension of (layer_dims[l], 1)
            
    Use zero initialization.
    """
    
    v = {}
    
    L = len(layer_dims) # total number of layers, including the input layer. 
    
    for l in range(1, L):
        v['dW' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        v['db' + str(l)] = np.zeros((layer_dims[l], 1))

    return v      




# initialize v and s for Adam optimization

def init_adam(layer_dims):
    """
    Arguments: 
    layer_dims: python array, layer_dims[l] is the number of units in the lth layer. 
                l = 0 is the input layer, the last l is the output layer.
    
    Returns:
    v: python dictionary containing velocities for each layer, 
            v['dW1'], v['dW2'], ..., v['dWl'], ..., v['db1'], v['db2'], ..., v['dbl'], ... 
            v['dWl'] has the dimension of (layer_dims[l], layer_dims[l - 1]).
            v['dbl'] has the dimension of (layer_dims[l], 1).
    s: python dictionary containing squared gradients for each layer, 
            s['dW1'], s['dW2'], ..., s['dWl'], ...,  s['db1'], s['db2'], ..., s['dbl'], ...  
            s['dWl'] has the dimension of (layer_dims[l], layer_dims[l - 1]).        
            s['dbl'] has the dimension of (layer_dims[l], 1)
            
    Use zero initialization.
    """
    
    v = {}
    s = {}
    
    L = len(layer_dims) # total number of layers, including the input layer. 
    
    for l in range(1, L):
        v['dW' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        v['db' + str(l)] = np.zeros((layer_dims[l], 1))
        s['dW' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        s['db' + str(l)] = np.zeros((layer_dims[l], 1))

    return v, s     




# create mini batches randomly, each with the size of mini_batch_size.

def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    """    
    Argument:
    X: input features, with dimension of (number of features, number of training examples)
    Y: true labels, the actual y values, with the dimension of (1, number of training examples)
    mini_batch_size: size of each mini batch.
    seed: random seed
    
    Returns:
    mini_batches: list of (mini_batch_X, mini_batch_Y)
    """
        
    np.random.seed(seed)
    
    m = X.shape[1] # total number of examples
    mini_batches = []
    
    # for batch gradient descent, no need to shuffle
    if mini_batch_size == X.shape[1]:
        mini_batch = (X, Y)
        mini_batches.append(mini_batch)
        return mini_batches
    
    # Step 1: shuffle
    permutation_indices = list(np.random.permutation(m))
    shuffled_X = X[:, permutation_indices]
    shuffled_Y = Y[:, permutation_indices]
    
    # Step 2: partition
    num_batch_except_last = math.floor(m / mini_batch_size)
    for i in range(num_batch_except_last):
        mini_batch_X = shuffled_X[:, i * mini_batch_size:(i + 1) * mini_batch_size - 1]
        mini_batch_Y = shuffled_Y[:, i * mini_batch_size:(i + 1) * mini_batch_size - 1]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    # last mini batch
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_batch_except_last * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_batch_except_last * mini_batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_Y))
        
    return mini_batches




# Calculate the Z value for forward propagation

def linear_forward_Z(A_prev, W, b):
    """
    Arguments:
    A_prev: the activations of the previous layer, with the dimension of (number of units in the previous layer, number of training examples)
    W: weight matrix for current layer
    b: bias vector for current layer
    
    Returns: 
    Z: the input of current layer's activation function, also called pre-activation parameter, Z[l] = W[l]A[l - 1] + b[l],
        with the dimension of (number of units in current layer, number of training examples)
    linear_cache: (A_prev, W, b), stored for calculating backward propagation
    """
    
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    
    # verify the dimension correctness
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
    return Z, linear_cache    




# Calculate the activation function for current layer

def linear_forward_activation(A_prev, W, b, activation):
    """
    Arguments:
    A_prev: the activations of the previous layer, with the dimension of (number of units in the previous layer, number of training examples)
    W: weight matrix for current layer
    b: bias vector for current layer
    activation: "sigmoid" or "relu"
    
    Returns:
    A: the output of current layer's activation function, also called post-activation value,
        with the dimension of (number of units in current layer, number of training examples)
    cache: ((A_prev, W, b), Z) stored for calculating backward propagation
    """
    
    Z, linear_cache = linear_forward_Z(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
        
    cache = (linear_cache, Z)
    
    return A, cache   




# whole forward propagation of L-layer deep neural network, without dropout.
# The output layer uses sigmoid activation, other layers use relu activation.
# L includes the input layer

def L_layer_forward(X, params):
    """
    Arguments:
    X: input features, with dimension of (number of features, number of training examples)
    params: python dictionary containing weight matrices wl and bias vectors bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
            
    Returns:
    Aout: output activation, i.e., the predicted y value, with the dimension of (1, number of examples)
    caches: the list of cache from linear_forward_activation(A_prev, W, b, activation) (totally L - 1 of them since L counts the input layer, 
    indexed from 0 to L - 2)
    """
    
    caches = []
    
    # Division (/) always returns a float. To do floor division and get an integer result (discarding any fractional result), 
    # you can use the // operator. 
    L = len(params) // 2 + 1 # total number of layers including the input layer
    
    A_prev = X
    
    # relu activation for the layers except the last one
    for l in range(1, L - 1):
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        A, cache = linear_forward_activation(A_prev, W, b, "relu")     
        caches.append(cache)        
        A_prev = A
        
    # sigmoid activation for the last layer
    Aout, cache = linear_forward_activation(A_prev, params['W' + str(L - 1)], params['b' + str(L - 1)], "sigmoid")
    caches.append(cache)
    
    assert(Aout.shape == (1, X.shape[1]))
    
    return Aout, caches




# whole forward propagation of L-layer deep neural network, with inverted dropout.
# The output layer uses sigmoid activation, other layers use relu activation.
# L includes the input layer

def L_layer_forward_inverted_dropout(X, params, keep_prob, seed = 0):
    """
    Arguments:
    X: input features, with dimension of (number of features, number of training examples)
    params: python dictionary containing weight matrices wl and bias vectors bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
    keep_prob: parameter for inverted dropout,the probability of keeping a neuron
    seed: random seed
    
    Returns:
    Aout: output activation, i.e., the predicted y value, with the dimension of (1, number of examples)
    caches: the list of cache from linear_forward_activation(A_prev, W, b, activation) (totally L - 1 of them since L counts the input layer, 
    indexed from 0 to L - 2)
    dropout_masks: the list of mask matrices for shutting down neurons in each layer (except the input and output layers)
    """
    
    np.random.seed(seed)
    
    caches = []
    dropout_masks = []
    
    L = len(params) // 2 + 1 # total number of layers including the input layer
    
    A_prev = X
    
    # relu activation for the layers except the last one
    for l in range(1, L - 1):
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        A, cache = linear_forward_activation(A_prev, W, b, "relu")
        caches.append(cache)        
         
        #mask = np.random.rand(A.shape[0], A.shape[1]) # random samples from a uniform distribution over [0, 1)
        #If you want an interface that takes a shape-tuple as the first argument, refer to np.random.random_sample
        mask = np.random.random_sample(A.shape)
        
        # to set all the entries of a matrix X to 0 (if entry is less than 0.4) or 1 (if entry is more than 0.4) you would do: X = (X < 0.4).
        # Note that 0 and 1 are respectively equivalent to False and True.
        mask = (mask < keep_prob)
        A = A * mask
        A = A / keep_prob
        dropout_masks.append(mask)
        
        A_prev = A
        
    # sigmoid activation for the last layer
    Aout, cache = linear_forward_activation(A_prev, params['W' + str(L - 1)], params['b' + str(L - 1)], "sigmoid")
    caches.append(cache)
    
    assert(Aout.shape == (1, X.shape[1]))
    
    return Aout, caches, dropout_masks     




# calculate the cross-entropy cost function

def cost_func(Aout, Y):
    """
    Arguments:
    Aout: output activation, i.e., the predicted y value, with the dimension of (1, number of examples)
    Y: true labels, the actual y values, with the dimension of (1, number of examples)
    
    Returns:
    cost: the cross-entropy cost
    """
    
    m = Y.shape[1]
    
    # numpy.multiply(): multiply arguments element-wise.Equivalent to x1 * x2 in terms of array broadcasting.
    # numpy.nansum(): treating Not a Numbers (NaNs) as zero.
    cost = -1 / m * np.nansum(np.multiply(Y, np.log(Aout)) + np.multiply(1 - Y, np.log(1 - Aout)), axis = 1, keepdims = True)
    cost = np.squeeze(cost) # e.g. this turns [[5]] into 5
    assert(cost.shape == ())
    
    return cost




# calculate the cost function with L2 regularization.

def cost_func_L2_Regul(Aout, Y, params, lambd):
    """
    Arguments:
    Aout: output activation, i.e., the predicted y value, with the dimension of (1, number of examples)
    Y: true labels, the actual y values, with the dimension of (1, number of training examples)
    params: python dictionary containing weight matrices wl and bias vectors bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
    lambd: the lambda parameter for L2 regularization
    
    Returns:
    cost: the cost value
    """
    
    m = Y.shape[1]    
    cross_entropy_cost = -1 / m * np.nansum(np.multiply(Y, np.log(Aout)) + np.multiply(1 - Y, np.log(1 - Aout)), axis = 1, keepdims = True)
    
    L = len(params) // 2 + 1
    L2_regularization_cost = 0 
    for l in range(1, L):
        L2_regularization_cost += np.nansum(np.square(params['W' + str(l)]))
    L2_regularization_cost *= lambd / (2 * m)
    
    cost = cross_entropy_cost + L2_regularization_cost   
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost




# backward propagation for current layer, given dZ
# In our calculation, dA and dZ are alwarys the derivatives with respect to the loss function NOT the cost function. 
# We treat dA and dZ differently from dW and dB. Namely, dA and dZ are always computing the derivatives dL/dA and dL/dZ respectively, 
# but dW and db are computing the derivatives dJ/dW and dJ/db.
# This is why we don't have the 1/m factor in the dA formula

def linear_backward_from_dZ(dZ, linear_cache):
    """
    Arguments:
    dZ: gradient of the loss with respect to Z for current layer
    linear_cache: (A_prev, W, b) stored from linear forward propagation for current layer
    
    Returns:
    dA_prev: gradient of the loss with respect to A_prev, for current layer, with the same dimension of A_prev
    dW: gradient of the cost with respect to W, for current layer, with the same dimension of W
    db: gradient of the cost with respect to b, for current layer, with the same dimension of b 
    """
    
    A_prev, W, b = linear_cache
    m = dZ.shape[1]
    
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev, dW, db




# backward propagation for current layer, given dZ, with L2 regularization

def linear_backward_from_dZ_L2_Regul(dZ, linear_cache, lambd):
    """
    Arguments:
    dZ: gradient of the loss with respect to Z for current layer
    linear_cache: (A_prev, W, b) stored from forward propagation for current layer
    lambd: the lambda parameter for L2 regularization
    
    Returns:
    dA_prev: gradient of the loss with respect to A_prev, for current layer, with the same dimension of A_prev
    dW: gradient of the cost with respect to W, for current layer, with the same dimension of W
    db: gradient of the cost with respect to b, for current layer, with the same dimension of b 
    """
    
    A_prev, W, b = linear_cache
    m = dZ.shape[1]
    
    dW = 1 / m * np.dot(dZ, A_prev.T) + (lambd / m) * W # this is the only difference between L2 and non-regularized
    db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev, dW, db




# backward propagation for current layer, given dA

def activation_backward_from_dA(dA, cache, activation):
    """
    Arguments:
    dA: gradient of the loss with respect to A[l] for current layer l
    cache: ((A_prev, W, b), Z) stored from forward propagation for current layer
    activation: "sigmoid" or "relu"
    
    Returns:
    dA_prev: gradient of the loss with respect to A_prev (A[l - 1]), for current layer, with the same dimension of A_prev
    dW: gradient of the cost with respect to W, for current layer, with the same dimension of W
    db: gradient of the cost with respect to b, for current layer, with the same dimension of b 
    """
    linear_cache, Z = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif activation == "relu":
        dZ = relu_backward(dA, Z)
    
    dA_prev, dW, db = linear_backward_from_dZ(dZ, linear_cache)
    
    return dA_prev, dW, db




# backward propagation for current layer, given dA, with L2 regularization

def activation_backward_from_dA_L2_Regul(dA, cache, activation, lambd):
    """
    Arguments:
    dA: gradient of the loss with respect to A[l] for current layer l
    cache: ((A_prev, W, b), Z) stored from forward propagation for current layer
    activation: "sigmoid" or "relu"
    lambd: the lambda parameter for L2 regularization
    
    Returns:
    dA_prev: gradient of the loss with respect to A_prev (A[l - 1]), for current layer, with the same dimension of A_prev
    dW: gradient of the cost with respect to W, for current layer, with the same dimension of W
    db: gradient of the cost with respect to b, for current layer, with the same dimension of b 
    """
    linear_cache, Z = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif activation == "relu":
        dZ = relu_backward(dA, Z)
    
    dA_prev, dW, db = linear_backward_from_dZ_L2_Regul(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db




# whole backward propagation of L-layer deep neural network
# The output layer use sigmoid activation, other layers use relu activation.

def L_layer_backward(Aout, Y, caches):
    """
    Arguments:
    Aout: output activation, i.e., the predicted y value, with the dimension of (1, number of examples)
    Y: true labels, the actual y values, with the dimension of (1, number of training examples)
    caches: the list of cache from L_layer_forward(X, params), i.e., list of ((A_prev, W, b), Z) 
            (totally L - 1 of them since L counts the input layer, indexed from 0 to L - 2)
            
    Returns: 
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['dbl'], ... 
    """
    
    grads = {}
    L = len(caches) + 1 # total number of layers including the input layer
    # gradient of the loss with respect to the output Aout
    # numpy.divide(): Divide arguments element-wise. Equivalent to x1 / x2 in terms of array-broadcasting.
    #dAout = - Y / Aout + (1 - Y) / (1 - Aout)
    dZout = Aout - Y    
    
    # calculate gradients for the last layer using sigmoid activation
    cache = caches[-1]
    linear_cache, _ = cache
    #dA_prev, dW, db = activation_backward_from_dA(dAout, cache, "sigmoid")   
    dA_prev, dW, db = linear_backward_from_dZ(dZout, linear_cache) 
    grads['dW' + str(L - 1)] = dW
    grads['db' + str(L - 1)] = db
 
    # calculate gradients for all the previous layers using relu activation
    for l in range(L - 3, -1, -1):
        cache = caches[l]
        dA_prev, dW, db = activation_backward_from_dA(dA_prev, cache, "relu")
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
        
    return grads




# whole backward propagation of L-layer deep neural network, with L2 regularization
# The output layer use sigmoid activation, other layers use relu activation.

def L_layer_backward_L2_Regul(Aout, Y, caches, lambd):
    """
    Arguments:
    Aout: output activation, i.e., the predicted y value, with the dimension of (1, number of examples)
    Y: true labels, the actual y values, with the dimension of (1, number of training examples)
    caches: the list of cache from L_layer_forward(X, params), i.e., list of ((A_prev, W, b), Z) 
            (totally L - 1 of them since L counts the input layer, indexed from 0 to L - 2)
    lambd: the lambda parameter for L2 regularization
            
    Returns: 
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['dbl'], ... 
    """
    
    grads = {}
    L = len(caches) + 1 # total number of layers including the input layer
    # gradient of the cost with respect to the output Aout
    # dAout = - Y / Aout + (1 - Y) / (1 - Aout)
    dZout = Aout - Y   
    
    # calculate gradients for the last layer using sigmoid activation
    cache = caches[-1]
    linear_cache, _ = cache
    #dA_prev, dW, db = activation_backward_from_dA_L2_Regul(dAout, cache, "sigmoid", lambd) 
    dA_prev, dW, db = linear_backward_from_dZ_L2_Regul(dZout, linear_cache, lambd) 
    grads['dW' + str(L - 1)] = dW
    grads['db' + str(L - 1)] = db
 
    # calculate gradients for all the previous layers using relu activation
    for l in range(L - 3, -1, -1):
        cache = caches[l]
        dA_prev, dW, db = activation_backward_from_dA_L2_Regul(dA_prev, cache, "relu", lambd) 
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
        
    return grads




# whole backward propagation of L-layer deep neural network, with inverted dropout
# The output layer use sigmoid activation, other layers use relu activation.

def L_layer_backward_inverted_dropout(Aout, Y, caches, keep_prob, dropout_masks):
    """
    Arguments:
    Aout: output activation, i.e., the predicted y value, with the dimension of (1, number of examples)
    Y: true labels, the actual y values, with the dimension of (1, number of training examples)
    caches: the list of cache from L_layer_forward(X, params), i.e., list of ((A_prev, W, b), Z) 
            (totally L - 1 of them since L counts the input layer, indexed from 0 to L - 2)
    keep_prob: parameter for inverted dropout,the probability of keeping a neuron
    dropout_masks: the mask matrices for shutting down neurons in each layer (except the input and output layers)
            
    Returns: 
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['dbl'], ... 
    """
    
    grads = {}
    L = len(caches) + 1 # total number of layers including the input layer
    # gradient of the cost with respect to the output Aout
    # dAout = - Y / Aout + (1 - Y) / (1 - Aout)
    dZout = Aout - Y 
    
    # calculate gradients for the last layer using sigmoid activation
    cache = caches[-1]
    linear_cache, _ = cache
    # dA_prev, dW, db = activation_backward_from_dA(dAout, cache, "sigmoid")
    dA_prev, dW, db = linear_backward_from_dZ(dZout, linear_cache) 
    grads['dW' + str(L - 1)] = dW
    grads['db' + str(L - 1)] = db
 
    # calculate gradients for all the previous layers using relu activation
    for l in range(L - 3, -1, -1):
        cache = caches[l]
        
        mask = dropout_masks[l]
        assert(mask.shape == dA_prev.shape)
        dA_prev = mask * dA_prev
        dA_prev = dA_prev / keep_prob
        
        dA_prev, dW, db = activation_backward_from_dA(dA_prev, cache, "relu")
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
        
    return grads




# Update the parameters including weight matrices and bias vectors

def update_params(params, grads, learning_rate):
    """
    Arguments:
    params: python dictionary containing weight matrix wl and bias vector bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['dbl'], ... 
    learning_rate: learning rate alpha
    
    Returns:
    updated params            
    """
    L = len(params) // 2 + 1 # total number of layers including the input layer
    
    for l in range(1, L):
        params['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * grads['db' + str(l)]
        
    return params    




# Update the parameters including weight matrices and bias vectors, with Momentum optimization

def update_params_momentum(params, v, grads, learning_rate, beta):
    """
    Arguments:
    params: python dictionary containing weight matrix wl and bias vector bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
    v: python dictionary containing velocities for each layer, 
            v['dW1'], v['dW2'], ..., v['dWl'], ..., v['db1'], v['db2'], ..., v['dbl'], ... 
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['dbl'], ... 
    learning_rate: learning rate alpha
    beta: hyperparameter for momentum optimization
    
    Returns:
    updated params and velocities            
    """
    L = len(params) // 2 + 1 # total number of layers including the input layer
    
    for l in range(1, L):
        v['dW' + str(l)] = beta * v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
        params['W' + str(l)] -= learning_rate * v['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * v['db' + str(l)]
        
    return params, v    




# Update the parameters including weight matrices and bias vectors, with Adam optimization

def update_params_adam(params, v, s, t, grads, learning_rate, beta1, beta2, epsilon):
    """
    Arguments:
    params: python dictionary containing weight matrix wl and bias vector bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
    v: python dictionary containing velocities for each layer, 
            v['dW1'], v['dW2'], ..., v['dWl'], ..., v['db1'], v['db2'], ..., v['dbl'], ... 
    s: python dictionary containing squared gradients for each layer, 
            s['dW1'], s['dW2'], ..., s['dWl'], ..., s['db1'], s['db2'], ..., s['dbl'], ...
    t: current number of steps (starting from 1)
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['bl'], ... 
    learning_rate: learning rate alpha
    beta1, beta2: hyperparameters for Adam optimization
    epsilon: hyperparameter preventing division by zero in Adam updates
    
    Returns:
    updated params, exponentially weighted average of the past gradients and squares of the past gradients      
    """
    L = len(params) // 2 + 1 # total number of layers including the input layer
    v_corrected = {}
    s_corrected = {}
    
    for l in range(1, L):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - beta1 ** t)
        
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - beta1 ** t)
        
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * grads['dW' + str(l)] ** 2
        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - beta2 ** t)
        
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * grads['db' + str(l)] ** 2
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - beta2 ** t)
        
        params['W' + str(l)] -= learning_rate * v_corrected['dW' + str(l)] / (np.sqrt(s_corrected['dW' + str(l)]) + epsilon)
        params['b' + str(l)] -= learning_rate * v_corrected['db' + str(l)] / (np.sqrt(s_corrected['db' + str(l)]) + epsilon)
       
    return params, v, s    




# Build the L-layer deep neural network model

def L_layer_model(X, Y, layer_dims, params_seed = 0, mini_batch_size = None, optimizer = "gd", learning_rate = 0.0075, beta = 0.9, 
                  beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, iterations = 3300, print_cost_freq = 0, save_cost_freq = 100, 
                  initialization = "he", regularization = "none", lambd = 0, keep_prob = 1, dropout_seed = 0):
    """
    Arguments:
    X: training set features, with the dimension of (number of features, number of examples)
    Y: training set labels, with the dimension of (1, number of examples)
    layer_dims: python array, layer_dims[l] is the number of units in the lth layer. 
                l = 0 is the input layer, the last l is the output layer
    params_seed: random seed for initialization.
    mini_batch_size: size of each mini batch. By default it is the total number of examples, i.e., batch gradient descent
    optimizer: choose the optimization method:
               "gd": gradient descent
               "momentum": momentum optimization
               "adam": Adam optimization
    learning_rate: learning rate for gradient descent
    beta: hyperparameter for momentum optimization
    beta1, beta2: hyperparameters for Adam optimization
    epsilon: hyperparameter preventing division by zero in Adam updates
    iterations: number of iterations
    print_cost_freq: if > 0, print the cost value every print_cost_freq steps.
    save_cost_freq: save the cost value every save_cost_freq steps into costs, for ploting the learning curve.
    initialization: choose which initialization to use: 
                "random_normal":  random values following standard normal distribution.
                "he": He initialization
                "xavier": Xavier initialization
    regularization: choose the regularization method:
                "none": no regularization
                "L2": L2 regularization
    lambd: the lambda parameter for L2 regularization
    keep_prob: keep_prob: parameter for inverted dropout,the probability of keeping a neuron. If it's 1, dropout is not used.
    dropout_seed: random seed for dropout.
    
    Returns:
    params: python dictionary containing weight matrix wl and bias vector bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
            Wl has the dimension of (layer_dims[l], layer_dims[l - 1]).
            bl has the dimension of (layer_dims[l], 1)
            
    Note: in this code, dropout and L2 regularization are not used at the same time
    """
    assert(keep_prob == 1 or regularization == "none")
    assert(keep_prob <= 1)
    
    if mini_batch_size is None:
        mini_batch_size = X.shape[1]
    
    costs = []
    seed_mini_batches = 10 # random seed for creating mini batches
    t = 0 # count current number of steps
    
    # initialization
    params = init_params(layer_dims, initialization, params_seed)
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = init_momentum(layer_dims)
    elif optimizer == "adam":
        v, s = init_adam(layer_dims)
    
    for i in range(1, iterations + 1):
        # create mini batches
        seed_mini_batches += 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed_mini_batches)
        
        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_Y = mini_batch
            
            # forward propagation
            if keep_prob < 1:
                Aout, caches, dropout_masks = L_layer_forward_inverted_dropout(mini_batch_X, params, keep_prob, dropout_seed)
            else:
                Aout, caches = L_layer_forward(mini_batch_X, params)
        
            # cost
            if regularization == "L2":
                cost = cost_func_L2_Regul(Aout, mini_batch_Y, params, lambd)
            else:
                cost = cost_func(Aout, mini_batch_Y)    
        
            # backward propagation
            if keep_prob < 1:
                grads = L_layer_backward_inverted_dropout(Aout, mini_batch_Y, caches, keep_prob, dropout_masks)
            elif regularization == "none":
                grads = L_layer_backward(Aout, mini_batch_Y, caches)
            elif regularization == "L2":
                grads = L_layer_backward_L2_Regul(Aout, mini_batch_Y, caches, lambd)
        
            # update weight matrices and bias vectors
            if optimizer == "gd":
                params = update_params(params, grads, learning_rate)
            elif optimizer == "momentum":
                params, v = update_params_momentum(params, v, grads, learning_rate, beta)
            elif optimizer == "adam":
                t += 1
                params, v, s = update_params_adam(params, v, s, t, grads, learning_rate, beta1, beta2, epsilon)
        
        if print_cost_freq > 0 and (i == 1 or i % print_cost_freq == 0):
            print("current iteration: " + str(i) + ", cost: " + str(cost))
        if save_cost_freq > 0 and (i == 1 or i % save_cost_freq == 0):
            costs.append(cost)
       
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations * ' + str(print_cost_freq))
    plt.title('learning rate = ' + str(learning_rate))
    plt.show()

    return params




# convert params to a vector

def params_to_vector(params):
    """
    Arguments:
    params: python dictionary containing weight matrix wl and bias vector bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
    
    Returns:
    theta: one-column vector by flattening and concatenating params['W1'], params['b1'], params['W2'], ..., 
            and params['b2'], ..., params['Wl'], params['bl'], ...
    keys: list of keys for each row of theta, ['W1', 'W1'...,'b1','b1',...]
    """
    keys = []
    L = len(params) // 2 + 1
    first_flag = True
    for l in range(1, L):
        cur_vector = np.reshape(params['W' + str(l)], (-1, 1))
        keys = keys + ['W' + str(l)] * cur_vector.shape[0]
        if first_flag:
            theta = cur_vector
            first_flag = False
        else:
            theta = np.concatenate((theta, cur_vector), axis = 0)
            
        cur_vector = np.reshape(params['b' + str(l)], (-1, 1))
        keys = keys + ['b' + str(l)] * cur_vector.shape[0]
        theta = np.concatenate((theta, cur_vector), axis = 0)
    
    return theta, keys




# convert grads to a vector

def grads_to_vector(grads):
    """
    Arguments:
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['dbl'], ... 
    
    Returns:
    theta: one-column vector by flattening and concatenating grads['dW1'], grads['db1'], grads['dW2'], ..., 
            and grads['db2'], ..., grads['dWl'], grads['dbl'], ...
    keys: list of keys for each row of theta, ['dW1', 'dW1'...,'db1','db1',...]
    """
    keys = []
    L = len(grads) // 2 + 1
    first_flag = True
    for l in range(1, L):
        cur_vector = np.reshape(grads['dW' + str(l)], (-1, 1))
        keys = keys + ['dW' + str(l)] * cur_vector.shape[0]
        if first_flag:
            theta = cur_vector
            first_flag = False
        else:
            theta = np.concatenate((theta, cur_vector), axis = 0)
            
        cur_vector = np.reshape(grads['db' + str(l)], (-1, 1))
        keys = keys + ['db' + str(l)] * cur_vector.shape[0]
        theta = np.concatenate((theta, cur_vector), axis = 0)
    
    return theta, keys
        




# convert vector back to params

def vector_to_params(theta, params):
    """
    Arguments:
    theta: one-column vector by flattening and concatenating grads['dW1'], grads['db1'], grads['dW2'], ..., 
            and grads['db2'], ..., grads['dWl'], grads['dbl'], ...
    params: original python dictionary containing weight matrix wl and bias vector bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
            it has the information of the exact dimension of each matrix
    
    Returns:
    params: converted weight matrices and bias vectors from theta
    """
    L = len(params) // 2 + 1
    index = 0;
    
    for l in range(1, L):
        params['W' + str(l)] = theta[index : index + params['W' + str(l)].size, 0].reshape(params['W' + str(l)].shape)
        index += params['W' + str(l)].size
        params['b' + str(l)] = theta[index : index + params['b' + str(l)].size, 0].reshape(params['b' + str(l)].shape)
        index += params['b' + str(l)].size
    
    return params




# gradient checking

def gradient_checking(params, grads, X, Y, epsilon = 1e-7):
    """
    Arguments:
    params: python dictionary containing weight matrix wl and bias vector bl for the lth layer, 
            params['W1'], params['W2'], ..., params['Wl'], ..., and params['b1'], params['b2'], ..., params['bl'], ... 
    grads: python dictionary containing gradients of the cost with respect to the weight matrices and bias vectors in each layer,
            grads['dW1'], grads['dW2'], ..., grads['dWl'], ..., and grads['db1'], grads['db2'], ..., grads['dbl'], ..., 
            to be compared to "gradapprox"
    X: data set features, with the dimension of (number of features, number of examples)
    Y: data set labels, with the dimension of (1, number of examples)
    
    Returns:
    difference: difference between the approximated gradient and the backward propagation gradient, defined above
    """
    params_values, _ = params_to_vector(params)
    grads_values, _ = grads_to_vector(grads)
    num_params = params_values.shape[0]
    J_plus = np.zeros(params_values.shape)
    J_minus = np.zeros(params_values.shape)
    
    for i in range(num_params):
        # get J_plus
        params_values_copy = np.copy(params_values)
        params_values_copy[i, 0] += epsilon
        Aout, _ = L_layer_forward(X, vector_to_params(params_values_copy, params.copy()))
        J_plus[i, 0] = cost_func(Aout, Y)
        # get J_minus
        params_values_copy = np.copy(params_values)
        params_values_copy[i, 0] -= epsilon
        Aout, _ = L_layer_forward(X, vector_to_params(params_values_copy, params.copy()))
        J_minus[i, 0] = cost_func(Aout, Y)
        
    # get approximated gradients
    gradapprox = np.subtract(J_plus, J_minus) / (2 * epsilon)
    
    # calculate difference
    difference = np.linalg.norm(grads_values - gradapprox) / (np.linalg.norm(grads_values) + np.linalg.norm(gradapprox))
    
    if difference > 2e-7:
        # font color:
        # Red = '\033[91m', Green = '\033[92m', Blue = '\033[94m', Cyan = '\033[96m', White = '\033[97m',
        # Yellow = '\033[93m', Magenta = '\033[95m', Grey = '\033[90m', Black = '\033[90m', Default = '\033[99m'
        # end: '\033[0m'
        print('\033[91m' + 'There is a mistake, the difference is ' + str(difference) + '\033[0m')
    else:
        print('\033[92m' + 'Good, the difference is ' + str(difference) + '\033[0m')
    
    return difference

