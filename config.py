import time

# Default configurations
#          UP   lEFT    DOWN   RIGHT
options = [38,  37,     40,     39]
op =      [87,  65,     83,     68]
traduction = {  op[0]: options[0], op[1]: options[1], op[2]:options[2], op[3]:options[3],
                options[0]: options[0], options[1]: options[1], options[2]:options[2], options[3]:options[3]}
#     DEL
undo = 8

LR = 1e-3
goal_steps = 20000
score_requirement = 4096
initial_games = 10000

_tick2_frame=0
_tick2_fps=20000000 # real raw FPS
_tick2_t0=time.time()
