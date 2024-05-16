# small language model

this is an nlp experiment to make a tiny chatbot :)

it is using a really simple four layer architecture which is made recurrent by concatenating the previous timestep's 2nd hidden layer with the current timestep's 1st input layer. the 1st hidden layer in the network can be seen like an embedding of the current token, and the 2nd hidden layer like an emvedding of the context, which is itself a function of the current token embedding and the previous context embedding.

the model is able to generate reasonable sentences over a small vocabulary with just a few hundred parameters :)

## training

make sure numpy is installed:
```
python3 -m pip install numpy
```
run the training process:
```
python3 train.py
```
after this the weights will be saved to `weights.npz`.

## inference

you can also run the network for different inputs like this, where a high output is a positive prediction:
```
$ python3 infer.py "This is a great movie, loved every second and will be watching again!"
[2.15298487]
$ python3 infer.py "I am not sure about this movie, maybe skip this one"
[-0.00422019]
```

