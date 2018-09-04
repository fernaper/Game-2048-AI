"""
https://github.com/yangshun/2048-python
"""

from tkinter import *
from logic import *
from random import *
import config as conf

SIZE = 500
GRID_LEN = 4

class Game():
    def __init__(self, score=0, first=True, matrix=None):
        self.score = score
        self.first = first
        self.commands = {conf.options[0]: up, conf.options[1]: left, conf.options[2]:down, conf.options[3]:right }

        if not matrix:
            self.init_matrix()
        else:
            self.matrix = matrix

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)
        self.matrix,_,_=add_two(self.matrix)
        self.matrix,_,_=add_two(self.matrix)

    def move(self, key):
        done = False
        key = conf.traduction[key]
        if key in self.commands:
            self.matrix,done,score = self.commands[key](self.matrix)
            self.score += score
            #print('Score: {}'.format(self.score))
            if done:
                self.matrix,_,_ = add_two(self.matrix)
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
