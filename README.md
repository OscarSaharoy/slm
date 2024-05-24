# small language model

this is an nlp experiment to make a tiny chatbot :)

it is using a really simple four layer architecture which is made recurrent by concatenating the previous timestep's 2nd hidden layer with the current timestep's 1st input layer. the 1st hidden layer in the network is like an embedding of the current token, and the 2nd hidden layer is like an embedding of the context, which is a function of the current token embedding and the previous context embedding.

the model is able to generate reasonable sentences over a very small vocabulary with just a few hundred parameters :) the training data makes it into a chatbot named boris who loves frogs.

## training

make sure numpy is installed:
```
python3 -m pip install numpy
```
run the training process:
```
python3 train.py
```
after this the weights will be saved to `weights.npz` and some sample output from the network will be printed.

## inference

you can run the network to complete prompts with the infer script like this:
```
$ ./infer.py "who are you?"
i am boris.
$ ./infer.py "what do you love?"
frogs.
```
it only knows a few words though which are in `wordmap.json` - other words are just converted to an unknown word token.

## embeddings

the hidden layers have semantic interpretations as embeddings of a word or whole sentence. to investigate this, the `searchembeddings.py` script will look for the words that are closest to an input word in embedding space by comparing the second hidden layer of the network when the words are fed through, and printing the 5 closest ones. you can see positive adjectives are close together, and so are question words.

```
$ ./searchembeddings.py "good"
[('good', 0.0), ('nice', 0.46020754596450925), ('.', 0.5950859535028892), ('is', 1.5625579068145068), ('thing', 1.681436310537393)]
$ ./searchembeddings.py "who"
[('who', 0.0), ('?', 1.3721200404159846), ('how', 1.6996846881430183), ('what', 2.1081211336680465), ('are', 2.6312298617361227)]
```

i also created a `compareembeddings.py` script that takes two sentences and calculates the distance between them in embedding space, and you can see the embeddings for similar sentences are closer together than those for opposing ones.

```
$ ./compareembeddings.py "i am good" "i am nice"
1.3235389881128539
$ ./compareembeddings.py "i am good" "i am bad"
1.9960487879227995
```

