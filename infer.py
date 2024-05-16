#!/usr/bin/env python3

import numpy as np
import json
import sys

# load dataset

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )

mapword = { v: k for k, v in wordmap.items() }
vocab_size = len(wordmap)

def relu( x ):
    return ( ( x > 0 ) + ( x < 0 ) * .01 ) * x
act = relu


# load wordmap

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )

input_size = 16
embedding_size = 10

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
    return np.exp(x) / np.sum(np.exp(x))
act = relu
dact = drelu

def predict( t, e_c_prev, weights ):
    w0, w1, w2 = weights
    e_t = act( w0 @ t )
    e_c = act( w1 @ np.concatenate(( e_t, e_c_prev )) )
    pred = softmax( w2 @ e_c )
    return pred, e_c

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

def complete( prompt, weights ):
    words = [ 0 ] + [
        wordmap[word]
        for word in re.findall(r"[\w']+|[.,!?;]", line)
        if word and word in wordmap and wordmap[word] < vocab_size
    ]

    e_c_prev = np.zeros(embedding_size)
    for word in words[:-1]:
        t = onehot(past_token)
        _, e_c_prev = predict( t, e_c_prev, weights )
    t = onehot( words[-1] )

    gen = np.random.default_rng()
    sentence = []

    while (len(sentence) == 0 or sentence[-1] != "") and len(sentence) < 100:
        pred, e_c_prev = predict( t, e_c_prev, weights )
        word = mapword[ gen.choice(input_size, p=pred) ]
        sentence.append(word)
        t = onehot( wordmap[word] )

    print(" ".join(sentence).replace(" .", "."))

# load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = [ f["w0"], f["w1"], f["w2"] ]


print( sum(w.size for w in weights), "parameters" )
for _ in range(20):
    write( weights )

if len( sys.argv ) > 1:
    print( complete( sys.argv[1], weights ) )
