import puzzle
import inv_puzzle
import random
import time
import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# MANUAL
#gamegrid = puzzle.GameGrid()
#gamegrid.mainloop()

def manual_move():
    gamegrid = puzzle.GameGrid()
    gamegrid.mainloop()

# IA
options = ["'w'","'a'","'s'","'d'"]
LR = 1e-3
goal_steps = 20000
score_requirement = 2048
initial_games = 10000

def sorted_prediction(prediction):
    return sorted(range(len(prediction)), key=lambda k: prediction[k], reverse=True)

def equals_grid(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def visual_ia_game(model):
    memory = []
    prev_obs = []
    choices = []
    gamegrid = puzzle.GameGrid()
    for _ in range(goal_steps):
        prev_obs = np.concatenate(gamegrid.matrix, axis=0)
        prediction = sorted_prediction(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])

        for i in range(len(prediction)):
            action = options[prediction[i]]
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

# We are goint to train it without any GUI becouse it is more efficient
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    i = 0
    while i < initial_games or len(accepted_scores) < initial_games*0.2:
        gamegrid = inv_puzzle.Game()
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            prev_observation = np.concatenate(gamegrid.matrix, axis=0)

            while equals_grid(np.concatenate(gamegrid.matrix, axis=0), prev_observation):
                action = random.choice(options)
                done = gamegrid.move(action)

            game_memory.append([prev_observation, action])
            score += gamegrid.score - score # How much we win with this move

            if done:
                break

        print ('Game: {}, Passed: {}/{}, Score: {}'.format(i+1, len(accepted_scores), initial_games*0.2, score))

        if score >= score_requirement:
            accepted_scores.append(score)

            for data in game_memory:
                if data[1] == "'w'":
                    output = [1,0,0,0]
                elif data[1] == "'a'":
                    output = [0,1,0,0]
                elif data[1] == "'s'":
                    output = [0,0,1,0]
                elif data[1] == "'d'":
                    output = [0,0,0,1]
                training_data.append([data[0], output])

        scores.append(score)
        i += 1

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

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
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='Game2048')

    return model


if __name__ == "__main__":
    #manual_move()
    #training_data = initial_population()
    #model = train_model(training_data)
    #model.save('basic.model')

    model = neural_network_model(16)
    model.load('basic.model')
    scores = []
    game_memory = []
    choices = []

    '''
    model = neural_network_model(264707)
    model.load('basic.model')
    '''

    for each_game in range(10):
        score, memory, choices = visual_ia_game(model)
        scores.append(score)
        game_memory.append(memory)
        choices.append(choices)

    print('Average score: ', sum(scores)/len(scores))
    print('w: {}%, a: {}%, s: {}%, d: {}%'.format(
        round(choices.count("'w'")/len(choices)*100,2),
        round(choices.count("'a'")/len(choices)*100,2),
        round(choices.count("'s'")/len(choices)*100,2),
        round(choices.count("'d'")/len(choices)*100,2)
    ))
    #some_random_games()
    #first_basic_moves()
    #manual_move()
