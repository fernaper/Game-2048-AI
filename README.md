![Game 2048](https://image.ibb.co/g1GrCz/2048_logo.png)

# Game 2048 AI #

This is an artificial intelligence project. Pretend using AI techniques to play the 2048's game.
It is based on training through simple heuristics of road exploration.

It is based on the training of a neural network (thanks tflearn) by means of simple heuristics of road exploration.

The results obtained are:

 <working on it>

# Pre-requisites

```
pip3 install -r requirements.txt
```

## How to train: ##

- Execute: `python3 Game2048 train`
- First it will generate training games with an exploration algorithm.
- You could stop this games when you think (but the default amount of games are a really good option).
- When it generates all the training data, it start training the neural network.

Note: If you want to re-train your neural network without generating more training games just execute: `python3 Game2048 train --games=0`

## How to test: ##

- Execute `python3 Game2048 play`
- If you want to see more games just change the command to: `python3 Game2048 play --games=10`

## How to play the game manually: ##

 - Execute `python3 Game2048 play --heuristic=manual`

## Credits: ##

 The base code of the Game (not the artificial intelligence) was made by [yangshun](https://github.com/yangshun/2048-python).
 I made some changes on the code, mainly to adapt it to play and train the neuronal network.

 The rest of the code has been created entirely by Fernando Pérez (@fernaperg).

## LICENSE: ##

 This code is free to use, but look carefully at the [LICENSE](LICENSE) file.
