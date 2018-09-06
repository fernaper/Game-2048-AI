# Default configurations
#          UP   lEFT    DOWN   RIGHT
options = [38,  37,     40,     39]
op =      [87,  65,     83,     68]
traduction = {  op[0]: options[0], op[1]: options[1], op[2]:options[2], op[3]:options[3],
                options[0]: options[0], options[1]: options[1], options[2]:options[2], options[3]:options[3]}
#           DEL
undo =      8

#           ESC
end_game =  27

LR = 1e-3
goal_steps = 20000
score_requirement = 8192
initial_games = 2000
