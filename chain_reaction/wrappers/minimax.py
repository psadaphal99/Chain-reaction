# The On Init and Init sections contain variables and functors,
# which will be modified by init function depending on backends.
# The sole reason of their existance is to keep the linter happy.


import random
import chain_reaction.backends.python.minimax_agent as pagent
import matplotlib.pyplot as plt
import numpy as np
import time
# ---------- ON INIT ---------------
load_scores = None
done = True
time_diff = []
game_count=0
game_iterations=[]
# ----------- INIT -----------------
def init(backend: str):

    global load_scores

    load_scores = pagent.load_scores


# ------- WRAPPER FUNCTIONS --------
def best_move(board: list, player: int, depth: int, randn: int, score0: list, score1: list) -> int:
    """
    Get weighted random choice of best n moves
    If there is an immediate winning move, always return it
    """
    global done
    global time_diff
    global game_count
    global game_iterations
    
    
    start_time = time.time()
    # make a list of (move, score)
    score_list = load_scores(board, player, depth)
    heatmap = list(enumerate(score_list))
    print(heatmap)

    data = np.random.rand(5, 5)
    c=0
    for i in range(5):
        for j in range(5):
            if(heatmap[c][1]>0):
                data[i][j]=heatmap[c][1]*1000
            else:
                data[i][j]= 0#heatmap[c][1]
            c=c+1

    # get random move with decreasing weights
    heatmap.sort(key=lambda x: x[1], reverse=True)
    m_moves = [i[0] for i in heatmap if i[1] > 0][:randn]
    weights = [6, 4, 2, 1, 1][: len(m_moves)]
    end_time = time.time()
    
    
    time_difference_ms = (end_time - start_time) * 1000
    time_diff.append(time_difference_ms)
    game_count=game_count+1
    game_iterations.append(game_count)
    
    
    # if there is a winning move or no random choice return
    # return random choice if more than one score is positive
    s1= pagent.board_score(board, player)
    s0= pagent.board_score(board, 1-player)
    score0.append(s0)
    score1.append(s1)
    print("player is", player)
    print("score of the player is = ", pagent.board_score(board, player) )
    print("score of the other player is = ", pagent.board_score(board, 1-player))
    print("")

    if heatmap[0][1] >= 41 and done:
        plt.clf()
        plt.imshow(data, cmap='autumn') 
        plt.title( "2-D Heat Map" )
        plt.savefig('heatmap.png')
        done=False
        plt.show()
        plt.clf()
        # plt.plot(game_iterations, time_diff)
        # plt.xticks(range(min(game_iterations), max(game_iterations)+1, 5))
        # plt.xlabel('game iterations label')
        # plt.ylabel('minimax time label')
        # plt.title('mimimax time vs game iterations Chart Example')
        # plt.savefig('minimax_time.png')
        # plt.show()
        plt.clf()

    if heatmap[0][1] == 10000 or len(m_moves) <= 1:
        print("score is = ", heatmap[0][1]) 
        return heatmap[0][0]
    else:
        return random.choices(m_moves, weights)[0]
