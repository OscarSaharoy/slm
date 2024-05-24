#!/usr/bin/env python3

import numpy as np
import json
import sys
import re

from train import input_size, embedding_size, vocab_size
from train import wordmap, mapword, act, softmax, predict, write, onehot


def search( word, weights ):

    t = onehot( wordmap.get( word, 1 ) )
    e_c_prev = np.zeros( embedding_size )

    _, embedding, _ = predict( t, e_c_prev, weights )

    closenesses = []

    with open("wordmap.json", "r") as f:
        for word in json.load(f).keys():
            t = onehot( wordmap.get( word, 1 ) )
            _, otherembedding, _ = predict( t, e_c_prev, weights )
            closenesses.append(( word, np.linalg.norm( embedding - otherembedding ) ))

    return sorted(closenesses, key=lambda x: x[1] )[:5]


# load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = [ f["w0"], f["w1"], f["w2"] ]

if len( sys.argv ) > 1:
    print( search( sys.argv[1], weights ) )
else:
    print("Error: please provide an input word to search for embeddings around!")

