#!/usr/bin/env python3

import numpy as np

"""

t is the input token
e_t is the embedding of the input token
e_c_prev is the embedding of the previous context
e_c is the embedding of the new context

e_t = f(t)
one hot to token embedding
100 -> 10

e_c = f( [e_t, e_c_prev] )
double stacked token and previous context embedding to context embedding
20 -> 10

pred = f(e_c)
context embedding to one hot ish
10 -> 100

"""

input_size = 100
embedding_size = 10
hidden_size = 10


def relu( x ):
    return ( ( x > 0 ) + ( x < 0 ) * .01 ) * x
def drelu( x ):
    return ( x > 0 ) + ( x < 0 ) * .01
act = relu
dact = drelu

def predict( t, e_c_prev, weights ):
    w1, w2, w3 = weights
    e_t = act( w1 @ t )
    e_c = act( w2 @ np.concatenate( e_t, e_c_prev ) )
    pred = act( w3 @ e_c )
    return pred

def loss( t, e_c_prev, weights, target ):
    pureloss = predict( t, e_c_prev, weights ) - target
    return pureloss.T @ pureloss

def check( t, e_c_prev, weights, target ):
    return np.argmax( predict( t, e_c_prev, weights ) ) == np.argmax( target )

def test_eval( t, e_c_prev, weights, targets ):
    return sum(
        check( t, e_c_prev, weights, target )
        for t, target in zip(obss, targets)
    ) / targets.shape[0]

def dldw( t, e_c_prev, weights, target ):
    w1, w2, w3 = weights

    z_hid = w1 @ obs
    hid = act( z_hid )
    z_out = w2 @ hid
    pred = act( z_out )

    z_e_t = w1 @ t
    e_t = act( z_e_t )
    z_e_c = w2 @ np.concatenate( e_t, e_c_prev )
    e_c = act( z_e_c )
    z_pred = w3 @ e_c
    pred = act( z_pred )

    error_pred = 2 * ( pred - target ) * dact(z_out)
    error_e_c = w3.T @ error_pred * dact(z_e_c)
    error_e_t = w2.T @ error_e_c * dact(z_e_t)

    dlossdw3 = np.outer( error_pred, e_c )
    dlossdw2 = np.outer( error_e_c, e_t )
    dlossdw1 = np.outer( error_e_t, t )

    return dlossdw1, dlossdw2, dlossdw3

# load dataset

with np.load( "mnist.npz", allow_pickle=True ) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]

# training set

obss = x_train[:10000].reshape( 10000, 28*28 ) / 256
targets = np.zeros((10000, 10))
targets[np.arange(10000), y_train[:10000]] = 1

# test set

obss_test = x_test[:1000].reshape( 1000, 28*28 ) / 256
targets_test = np.zeros((1000, 10))
targets_test[np.arange(1000), y_test[:1000]] = 1

# init network

np.random.seed(1)
w1 = np.random.rand(40, 28*28) - .5
w2 = np.random.rand(10, 40) - .5
w1 *= .2
w2 *= .2
weights = [ w1, w2 ]
a = 0.005

# training loop

try:
    for epoch in range(11):
        for i, obs in enumerate(obss):
            target = targets[i]
            dw1, dw2 = dldw( obs, weights, target )
            weights[0] -= a * dw1
            weights[1] -= a * dw2
        print(
            "epoch", epoch,
            "- test accuracy", f"{test_eval( obss_test, weights, targets_test ) * 100:.1f}%"
        )

except KeyboardInterrupt:
    print("ending training!")

# save the weights to a file

np.savez( "weights.npz", w1=weights[0], w2=weights[1] )

# to load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = [ f["w1"], f["w2"] ]
