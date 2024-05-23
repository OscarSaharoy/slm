#!/usr/bin/env python3

import numpy as np
import json
import sys
import re

from train import input_size, embedding_size, vocab_size
from train import wordmap, mapword, act, softmax, predict, write, onehot


def complete( prompt, weights ):
    words = [ 0 ] + [
        wordmap.get(word, 1)
        for word in re.findall(r"[\w']+|[-.,!?;]", prompt)
    ]

    e_c_prev = np.zeros(embedding_size)
    for word in words[:-1]:
        t = onehot(word)
        _, e_c_prev = predict( t, e_c_prev, weights )
    t = onehot( words[-1] )

    gen = np.random.default_rng()
    sentence = []

    while (len(sentence) == 0 or sentence[-1] != "") and len(sentence) < 100:
        pred, e_c_prev = predict( t, e_c_prev, weights )
        word = mapword[ gen.choice(input_size, p=pred) ]
        sentence.append(word)
        t = onehot( wordmap[word] )

    return " ".join(sentence).replace(" .", ".").replace(" ?", "?")

# load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = [ f["w0"], f["w1"], f["w2"] ]

if len( sys.argv ) > 1:
    print( complete( sys.argv[1], weights ) )
else:
    for _ in range(20):
        write( weights )
