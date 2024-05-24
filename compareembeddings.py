#!/usr/bin/env python3

import numpy as np
import json
import sys
import re

from train import input_size, embedding_size, vocab_size
from train import wordmap, mapword, act, softmax, predict, write, onehot


def embed( prompt, weights ):
    words = [ 0 ] + [
        wordmap.get(word, 1)
        for word in re.findall(r"[\w']+|[-.,!?;]", prompt)
    ]

    e_c_prev = np.zeros(embedding_size)
    for word in words:
        t = onehot(word)
        _, _, e_c_prev = predict( t, e_c_prev, weights )

    return e_c_prev

def compare( prompt1, prompt2, weights ):
    return np.linalg.norm(
        embed( prompt1, weights ) - embed( prompt2, weights )
    )

# load the weights

with np.load( "weights.npz", allow_pickle=True ) as f:
    weights = [ f["w0"], f["w1"], f["w2"] ]

if len( sys.argv ) > 2:
    print( compare( sys.argv[1], sys.argv[2], weights ) )
else:
    print("Error: please provide two input strings to compare embeddings for!")

