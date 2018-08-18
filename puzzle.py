"""
https://github.com/yangshun/2048-python
"""

from tkinter import *
from logic import *
from random import *
import config as conf

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e", 4096:"#3d3a31" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2", 4096:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")

class GameGrid(Frame):
    def __init__(self, visual=True):
        Frame.__init__(self)

        self.first = True
        self.score = 0
        self.is_end_game = False
        self.moved = False
        self.last_move = ''
        self.undo = False
        self.last_two = (0,0)
        self.visual = visual
        self.grid()
        self.master.title('2048 - Score: 0')
        self.master.bind("<Key>", self.key_down)

        #self.gamelogic = gamelogic
        self.commands = {conf.options[0]: up, conf.options[1]: left, conf.options[2]:down, conf.options[3]:right }

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        self.prev_matrix = self.matrix
        self.prev_score = self.score

        self.update_idletasks()
        self.update()

        #self.mainloop()

    def init_grid(self):
        if self.visual:
            background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
            background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                if self.visual:
                    cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                    cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                    # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                    t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                    t.grid()
                    grid_row.append(t)
            if self.visual:
                self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix,_=add_two(self.matrix)
        self.matrix,self.last_two=add_two(self.matrix)

    def update_grid_cells(self):
        if self.visual:
            for i in range(GRID_LEN):
                for j in range(GRID_LEN):
                    new_number = self.matrix[i][j]
                    if new_number == 0:
                        self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                    else:
                        n = new_number
                        if n > 4096:
                            n = 4096
                        self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[n], fg=CELL_COLOR_DICT[n])
            self.update_idletasks()

    def key_down(self, event):
        if event.keycode == conf.undo:
            if self.undo:
                print('Creating undo')
                self.matrix = self.prev_matrix
                self.score = self.prev_score
                self.undo = False
                self.update_grid_cells()
            return True

        if not event.keycode in conf.options and not event.keycode in conf.op:
            return False

        self.last_move = conf.traduction[event.keycode]
        return self.move(self.last_move)

    def move(self, key):
        prev_m = self.matrix
        prev_s = self.score

        if key in self.commands:
            self.matrix,done,score = self.commands[key](self.matrix)
            self.score += score
            self.master.title('2048 - Score: {}'.format(self.score))
            #print('Score: {}'.format(self.score))
            if done:
                if self.undo:
                    self.matrix, self.last_two = add_two(self.matrix)
                else:
                    self.matrix[self.last_two[0]][self.last_two[1]] = 2

                self.undo = True
                self.moved = True
                self.prev_matrix = prev_m
                self.prev_score = prev_s

                self.update_grid_cells()
                self.is_end_game=False
                if self.visual:
                    if self.first and game_state(self.matrix, self.first)=='win':
                        self.first = False
                        self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                        self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    if game_state(self.matrix, self.first)=='lose':
                        self.is_end_game = True
                        self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                        if self.first:
                            self.grid_cells[1][2].configure(text="Lose!",bg=BACKGROUND_COLOR_CELL_EMPTY)
                        else:
                            self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)

        if self.visual:
            self.update_idletasks()
            self.update()
        return self.is_end_game

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2
