# small language model

this is an nlp experiment to make a tiny chatbot :)

it is using a really simple four layer architecture which is made recurrent by concatenating the previous timestep's 2nd hidden layer with the current timestep's 1st input layer. the 1st hidden layer in the network is like an embedding of the current token, and the 2nd hidden layer is like an embedding of the context, which is a function of the current token embedding and the previous context embedding.

the model is able to generate reasonable sentences over a very small vocabulary with just a few hundred parameters :)

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
$ ./infer.py "i"
am nice.
$ ./infer.py "my dog"
is nice.
```
it only knows a few words though which are in `wordmap.json`.

