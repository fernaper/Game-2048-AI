"""
https://github.com/yangshun/2048-python
"""

from tkinter import *
from logic import *
from random import *

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"

class GameGrid(Frame):
    def __init__(self, visual=True):
        Frame.__init__(self)

        self.first = True
        self.score = 0
        self.visual = visual
        self.grid()
        self.master.title('2048 - Score: 0')
        self.master.bind("<Key>", self.key_down)

        #self.gamelogic = gamelogic
        self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

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

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_grid_cells(self):
        if self.visual:
            for i in range(GRID_LEN):
                for j in range(GRID_LEN):
                    new_number = self.matrix[i][j]
                    if new_number == 0:
                        self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                    else:
                        self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
            self.update_idletasks()

    def key_down(self, event):
        return self.move(repr(event.char))

    def move(self, key):
        done = False
        if key in self.commands:
            self.matrix,done,score = self.commands[key](self.matrix)
            self.score += score
            self.master.title('2048 - Score: {}'.format(self.score))
            #print('Score: {}'.format(self.score))
            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                done=False
                if self.visual:
                    if self.first and game_state(self.matrix, self.first)=='win':
                        self.first = False
                        self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                        self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    if game_state(self.matrix, self.first)=='lose':
                        done = True
                        self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                        if self.first:
                            self.grid_cells[1][2].configure(text="Lose!",bg=BACKGROUND_COLOR_CELL_EMPTY)
                        else:
                            self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)

        if self.visual:
            self.update_idletasks()
            self.update()
        return done

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2
