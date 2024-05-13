#!/usr/bin/env python3

import numpy as np
import json
import re

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

def onehot( x ):
    res = np.zeros(embedding_size)
    res[x] = 1
    return res

def relu( x ):
    return ( ( x > 0 ) + ( x < 0 ) * .01 ) * x
def drelu( x ):
    return ( x > 0 ) + ( x < 0 ) * .01
def softmax( x ):
    return np.exp(x) / np.sum(np.exp(x))
def dsoftmax( x ):
    return softmax(x) * ( 1 - softmax(x) )
act = relu
dact = drelu

def predict( t, e_c_prev, weights ):
    w0, w1, w2 = weights
    e_t = act( w0 @ t )
    e_c = act( w1 @ np.concatenate(( e_t, e_c_prev )) )
    pred = softmax( w2 @ e_c )
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
    pred = softmax( z_pred )

    error_pred = 2 * ( pred - target ) * dact(z_out)
    error_e_c = w3.T @ error_pred * dact(z_e_c)
    error_e_t = w2.T @ error_e_c * dact(z_e_t)

    dlossdw3 = np.outer( error_pred, e_c )
    dlossdw2 = np.outer( error_e_c, e_t )
    dlossdw1 = np.outer( error_e_t, t )

    return dlossdw1, dlossdw2, dlossdw3

np.random.seed(1)
a = 0.005
scale = .2

w0 = np.random.rand( embedding_size, input_size ) - .5
w1 = np.random.rand( embedding_size, embedding_size * 2 ) - .5
w2 = np.random.rand( input_size, embedding_size ) - .5
w0 *= scale
w1 *= scale
w2 *= scale
weights = [ w0, w1, w2 ]

# load dataset

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )

vocab_size = len(wordmap)
x = []

with open( "sentences.txt", 'r' ) as f:
    for line in f:
        words = [
            wordmap[word]
            for word in re.findall(r"[\w']+|[.,!?;]", line)
            if word and word in wordmap and wordmap[word] < vocab_size
        ] + [ 0 ]
        x.extend( [ (words[:i], elm) for i, elm in enumerate(words) ] )        

# train and test set

training_data = x[:len(x) // 10 * 9]
test_data = x[len(x) // 10 * 9:]

# training loop

try:
    for epoch in range(11):
        for past_tokens, next_token in training_data:
            e_c_prev = np.zeros(embedding_size)            
            t = onehot(past_tokens[-1])

            dw0, dw1, dw2 = dldw( obs, weights, target )
            weights[0] -= a * dw0
            weights[1] -= a * dw1
        print(
            "epoch", epoch,
            "- test accuracy", f"{test_eval( obss_test, weights, targets_test ) * 100:.1f}%"
        )

except KeyboardInterrupt:
    print("ending training!")

# save the weights to a file

np.savez( "weights.npz", w0=weights[0], w1=weights[1], w2=weights[2] )

# to load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = [ f["w0"], f["w1"], f["w2"] ]
