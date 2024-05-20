#!/usr/bin/env python3

import json

with open( "wordmap.json", mode="r" ) as f:
    wordmap = json.load( f )

print([ word for word in wordmap.keys() ])
