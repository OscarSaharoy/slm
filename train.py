#!/usr/bin/env python3

import numpy as np
import json
import re

"""

t is the input token
e_t is the embedding of the input token
e_c_prev is the embedding of the previous context
e_c is the embedding of the new context
tt is the target token

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

# load dataset

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )

mapword = { v: k for k, v in wordmap.items() }
vocab_size = len(wordmap)
x = []

with open( "sentences.txt", 'r' ) as f:
    for line in f:
        words = [ 0 ] + [
            wordmap[word]
            for word in re.findall(r"[\w']+|[.,!?;]", line)
            if word and word in wordmap and wordmap[word] < vocab_size
        ] + [ 0 ]
        x.append(words)

# train and test set

training_data = x[:len(x) // 10 * 9]
test_data = x[len(x) // 10 * 9:]

input_size = 16
embedding_size = 10
a = 0.05
scale = .2
epochs = 1000
l2 = 0.000

# funcs

def onehot( x ):
    res = np.zeros(input_size)
    res[x] = 1
    return res

def relu( x ):
    return ( ( x > 0 ) + ( x < 0 ) * .01 ) * x
def drelu( x ):
    return ( x > 0 ) + ( x < 0 ) * .01
def sigmoid( x ):
    return 1 / ( 1 + np.exp(-x) )
def dsigmoid( x ):
    return sigmoid(x) * ( 1 - sigmoid(x) )
def softmax( x ):
    cx = x - max(x)
    return np.exp(cx) / np.sum(np.exp(cx))
act = relu
dact = drelu

def predict( t, e_c_prev, weights ):
    w0, w1, w2 = weights
    e_t = act( w0 @ t )
    e_c = act( w1 @ np.concatenate(( e_t, e_c_prev )) )
    pred = softmax( w2 @ e_c )
    return pred, e_c

def loss( t, e_c_prev, weights, tt ):
    pureloss = predict( t, e_c_prev, weights )[0] - tt
    return pureloss.T @ pureloss

def check( t, e_c_prev, weights, tt ):
    return np.argmax( predict( t, e_c_prev, weights )[0] ) == np.argmax( tt )

def test_eval( test_data, weights ):
    res = 0
    for tokens in test_data:
        e_c_prev = np.zeros(embedding_size)
        for i, _ in enumerate( tokens[:-1] ):
            t = onehot( tokens[i] )
            tt = onehot( tokens[i+1] )
            pred, e_c_prev = predict( t, e_c_prev, weights )
            res += np.argmax( pred ) == np.argmax( tt )
    return res / sum( len(x) for x in test_data )

def dldwf( t, e_c_prev, weights, tt ):
    w1, w2, w3 = weights

    z_e_t = w1 @ t
    e_t = act( z_e_t )
    z_e_c = w2 @ np.concatenate(( e_t, e_c_prev ))
    e_c = act( z_e_c )
    z_pred = w3 @ e_c
    pred = softmax( z_pred )

    error_pred = 2 * ( pred - tt ) * pred * ( 1 - pred )
    error_e_c = w3.T @ error_pred * dact(z_e_c)
    error_e_t = ( w2.T @ error_e_c )[:embedding_size] * dact(z_e_t)
    error_e_c_prev = ( w2.T @ error_e_c )[embedding_size:] * dact(e_c_prev)

    dlossdw3 = np.outer( error_pred, e_c ) - l2 * 2 * w3
    dlossdw2 = np.outer( error_e_c, np.concatenate(( e_t, e_c_prev )) ) - l2 * 2 * w2
    dlossdw1 = np.outer( error_e_t, t ) - l2 * 2 * w1

    return dlossdw1, dlossdw2, dlossdw3, e_c, error_e_c_prev

def dldwi( t, e_c_prev, weights, error_e_c ):
    w1, w2, w3 = weights

    z_e_t = w1 @ t
    e_t = act( z_e_t )
    z_e_c = w2 @ np.concatenate(( e_t, e_c_prev ))
    e_c = act( z_e_c )

    error_e_t = ( w2.T @ error_e_c )[:embedding_size] * dact(z_e_t)
    error_e_c_prev = ( w2.T @ error_e_c )[embedding_size:] * dact(e_c_prev)

    dlossdw3 = 0
    dlossdw2 = np.outer( error_e_c, np.concatenate(( e_t, e_c_prev )) ) - l2 * 2 * w2
    dlossdw1 = np.outer( error_e_t, t ) - l2 * 2 * w1

    return dlossdw1, dlossdw2, dlossdw3, e_c, error_e_c_prev

def write( weights ):
    gen = np.random.default_rng()
    sentence = []
    e_c_prev = np.zeros(embedding_size)
    t = onehot(0)

    while (len(sentence) == 0 or sentence[-1] != "") and len(sentence) < 100:
        pred, e_c_prev = predict( t, e_c_prev, weights )
        word = mapword[ gen.choice(input_size, p=pred) ]
        sentence.append(word)
        t = onehot( wordmap[word] )

    print(" ".join(sentence).replace(" .", "."))

# init network

def train():
    np.random.seed(0)
    w0 = np.random.rand( embedding_size, input_size ) - .5
    w1 = np.random.rand( embedding_size, embedding_size * 2 ) - .5
    w2 = np.random.rand( input_size, embedding_size ) - .5
    w0 *= scale
    w1 *= scale
    w2 *= scale
    weights = [ w0, w1, w2 ]

    print( sum(w.size for w in weights), "parameters" )

    # training loop

    try:
        for epoch in range(epochs):
            for tokens in training_data:
                state_stack = []
                e_c_prev = np.zeros(embedding_size)
                for i, _ in enumerate( tokens[:-1] ):
                    t = onehot(tokens[i])
                    tt = onehot(tokens[i+1])
                    state_stack.append(( t, e_c_prev ))
                    dw0, dw1, dw2, e_c_prev, error_e_c_prev = \
                        dldwf( t, e_c_prev, weights, tt )
                    weights[0] -= a * dw0
                    weights[1] -= a * dw1
                    weights[2] -= a * dw2

                    for t_p, e_c_prev_p in state_stack[::-1]:
                        dw0, dw1, dw2, _, error_e_c_prev = \
                            dldwi( t_p, e_c_prev_p, weights, error_e_c_prev )
                        weights[0] -= a * dw0
                        weights[1] -= a * dw1
                        weights[2] -= a * dw2

            if epoch % 20 == 0:
                print(
                    "epoch", epoch,
                    "- test accuracy", f"{test_eval( test_data, weights ) * 100:.1f}%"
                )

    except KeyboardInterrupt:
        print("\nending training")

    print( sum(w.size for w in weights), "parameters" )
    for _ in range(20):
        write( weights )

    # save the weights to a file

    np.savez( "weights.npz", w0=weights[0], w1=weights[1], w2=weights[2] )

    # to load the weights

    with np.load( "weights.npz", allow_pickle=True ) as f:
        weights = [ f["w0"], f["w1"], f["w2"] ]

if __name__ == "__main__":
    train()
