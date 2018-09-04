import puzzle
import inv_puzzle
import random
import time
import gym
import numpy as np
import tflearn
import click
import copy

import config as conf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

@click.group()
def cli():
    pass

@cli.command('play')
@click.option('--heuristic', default='ia')
@click.option('--model', default='basic')
@click.option('--games', default=1)
def play(heuristic, model, games):
    if heuristic == 'ia':
        ia_move(model, True, games)
    elif heuristic == 'explorer':
        explorer_game()
    else:
        manual_move()

@cli.command('train')
@click.option('--model', default='basic')
@click.option('--games', default=conf.initial_games)
@click.option('--heuristic', default='random')
@click.option('--training_data', default='')
def train(model, games, heuristic, training_data):
    if training_data == '':
        training_data = initial_population(games, heuristic)
    else:
        training_data = np.load('saves/{}.npy'.format(training_data))
    m = train_model(training_data)
    m.save('models/{}.model'.format(model))

# MANUAL
def manual_move():
    gamegrid = puzzle.GameGrid()
    training_data = []
    game_memory = []
    prev_observation = []
    scores = []
    accepted_scores = []

    try:
        training_data = np.load('saves/manual.npy').tolist()
        print('Continuing manual training')
    except Exception as e:
        print('First manual training, good luck!')

    while not gamegrid.is_end_game:
        prev_observation = np.concatenate(gamegrid.matrix, axis=0)
        while not gamegrid.moved:
            gamegrid.update_idletasks()
            gamegrid.update()

        gamegrid.moved = False
        # Only if it is an active move, save it
        if not equals_grid(np.concatenate(gamegrid.matrix, axis=0), prev_observation):
            if not gamegrid.can_undo:
                # Redo this movement
                game_memory.pop()
            else:
                action = gamegrid.last_move
                game_memory.append([prev_observation, action])

    if gamegrid.score >= conf.score_requirement:
        accepted_scores.append(gamegrid.score)

        for data in game_memory:
            if data[1] == conf.options[0]:
                output = [1,0,0,0]
            elif data[1] == conf.options[1]:
                output = [0,1,0,0]
            elif data[1] == conf.options[2]:
                output = [0,0,1,0]
            elif data[1] == conf.options[3]:
                output = [0,0,0,1]
            training_data.append([data[0], output])
    scores.append(gamegrid.score)

    if training_data:
        np.save('saves/manual.npy',np.array(training_data))

# IA
def ia_move(model_name, load, games):
    if not load:
        training_data = initial_population(games)
        model = train_model(training_data)
        model.save('models/{}.model'.format(model_name))
    else:
        model = neural_network_model(16)
        try:
            model.load('models/{}.model'.format(model_name))
        except Exception as e:
            print('Model not found')
            return

    scores = []
    game_memory = []
    choices = []

    for each_game in range(games):
        score, memory, choices = visual_ia_game(model)
        scores.append(score)
        game_memory.append(memory)
        choices.append(choices)

    print('Average score: ', sum(scores)/len(scores))
    print('w: {}%, a: {}%, s: {}%, d: {}%'.format(
        round(choices.count(conf.options[0])/len(choices)*100,2),
        round(choices.count(conf.options[1])/len(choices)*100,2),
        round(choices.count(conf.options[2])/len(choices)*100,2),
        round(choices.count(conf.options[3])/len(choices)*100,2)
    ))

def equals_grid(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def sorted_prediction(prediction):
    return sorted(range(len(prediction)), key=lambda k: prediction[k], reverse=True)

# Game for the ia that you can see
def visual_ia_game(model):
    memory = []
    prev_obs = []
    choices = []
    gamegrid = puzzle.GameGrid()
    for _ in range(conf.goal_steps):
        prev_obs = np.concatenate(gamegrid.matrix, axis=0)
        prediction = sorted_prediction(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])

        for i in range(len(prediction)):
            action = conf.options[prediction[i]]
            done = gamegrid.move(action)

            if not equals_grid(np.concatenate(gamegrid.matrix, axis=0), prev_obs):
                break

        choices.append(action)
        memory = [gamegrid.matrix, action]

        gamegrid.update_idletasks()
        gamegrid.update()

        if done:
            break

        time.sleep(0.01)

    gamegrid.destroy()
    time.sleep(1)

    return gamegrid.score, memory, choices

def explorer_game():
    memory = []
    prev_obs = []
    choices = []
    gamegrid = puzzle.GameGrid()
    for _ in range(conf.goal_steps):
        action, _ = explorer_move(inv_puzzle.Game(gamegrid.score, gamegrid.first, gamegrid.matrix), conf.options, 5) # Max depth == 5
        done = gamegrid.move(action)

        choices.append(action)
        memory = [gamegrid.matrix, action]

        gamegrid.update_idletasks()
        gamegrid.update()

        if done:
            break

        time.sleep(0.01)

    gamegrid.destroy()
    time.sleep(1)

    return gamegrid.score, memory, choices

def corner_choice(matrix, options, invalid_moves):
    one_line_matrix = np.concatenate(matrix, axis=0)
    index_max_value = one_line_matrix.argmax(axis=0)

    line = index_max_value / len(matrix[0])
    column = index_max_value % len(matrix[0])

    if column == 0 and options[1] not in invalid_moves:
        if random.random() < 0.5:
            return options[1] # left
    if column == len(matrix[0])-1 and options[3] not in invalid_moves:
        if random.random() < 0.5:
            return options[3] # right
    if line == 0 and options[0] not in invalid_moves:
        if random.random() < 0.5:
            return options[0] # up
    if line == len(matrix[0]) and options[2] not in invalid_moves:
        if random.random() < 0.5:
            return options[2] # down

    move = [x for x in options if x not in invalid_moves]
    action = random.choice(move)
    return action

def left_down_corner_choice(matrix, options, invalid_moves):
    one_line_matrix = np.concatenate(matrix, axis=0)
    index_max_value = one_line_matrix.argmax(axis=0)

    line = index_max_value / len(matrix[0])
    column = index_max_value % len(matrix[0])

    if (column == 0 or random.random() < 0.5) and options[1] not in invalid_moves:
        return options[1] # left
    if (line == len(matrix[0]) or random.random() < 0.5) and options[2] not in invalid_moves:
        return options[2] # down
    move = [x for x in options if x not in invalid_moves]
    action = random.choice(move)
    return action

def explorer_move(gamegrid, options, depth):
    score = 0
    move = options[0]
    gg = np.concatenate(gamegrid.matrix, axis=0)

    for i in range(4):
        g = copy.copy(gamegrid)
        done = g.move(options[i])
        own_s = g.score - gamegrid.score
        # Our base case is when depth reached (0), finish the game or invalid move (the last one is just for truncate)
        if depth > 0 and not done and not equals_grid(gg, np.concatenate(g.matrix, axis=0)):
            m, s = explorer_move(g, options, depth - 1)
            own_s += s
        # Here we choose just the best option
        if own_s > score:
            score = own_s
            move = options[i]
    return move, score

# We are goint to train it without any GUI becouse it is more efficient
def initial_population(games, heuristic='random'):
    training_data = []
    scores = []
    accepted_scores = []
    i = 0
    print('Starting to train ...')
    while i < games or len(accepted_scores) < int(games*0.2):
        gamegrid = inv_puzzle.Game()
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(conf.goal_steps):
            prev_observation = np.concatenate(gamegrid.matrix, axis=0)
            invalid_moves = []
            while equals_grid(np.concatenate(gamegrid.matrix, axis=0), prev_observation):
                if heuristic == 'random':
                    move = [x for x in conf.options if x not in invalid_moves]
                    action = random.choice(move)
                elif heuristic == 'corner':
                    action = corner_choice(gamegrid.matrix, conf.options, invalid_moves)
                elif heuristic == 'one_corner':
                    action = left_down_corner_choice(gamegrid.matrix, conf.options, invalid_moves)
                elif heuristic == 'explorer':
                    action, s = explorer_move(gamegrid, conf.options, 2) # Max depth == 2

                invalid_moves.append(action)
                done = gamegrid.move(action)

            game_memory.append([prev_observation, action])
            score += gamegrid.score - score # How much we win with this move

            if done:
                break

        if score >= conf.score_requirement:
            accepted_scores.append(score)

            for data in game_memory:
                output = [0,0,0,0]
                output[conf.options.index(data[1])] = 1
                training_data.append([data[0], output])

        scores.append(score)
        i += 1
        print ('Game: {}, Passed: {}/{}, Score: {}'.format(i, len(accepted_scores), int(games*0.2), score))

    training_data_save = np.array(training_data)
    #np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

# The neural network model that uses the IA
def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 4, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=conf.LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

# Here we train the model
def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='Game2048')

    return model

if __name__ == "__main__":
    cli()
