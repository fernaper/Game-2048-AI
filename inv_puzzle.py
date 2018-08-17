"""
https://github.com/yangshun/2048-python
"""

from tkinter import *
from logic import *
from random import *

SIZE = 500
GRID_LEN = 4

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"

class Game():
    def __init__(self):
        self.score = 0
        self.first = True
        self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }

        self.grid_cells = []
        self.init_matrix()

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)
        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def move(self, key):
        done = False
        if key in self.commands:
            self.matrix,done,score = self.commands[key](self.matrix)
            self.score += score
            #print('Score: {}'.format(self.score))
            if done:
                self.matrix = add_two(self.matrix)
                done=False
                if self.first and game_state(self.matrix, self.first)=='win':
                    self.first = False
                if game_state(self.matrix, self.first)=='lose':
                    done = True
        return done

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

    def show(self):
        print(self.score)
        print(self.matrix)
