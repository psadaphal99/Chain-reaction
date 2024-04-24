import time
import math
import random

import chain_reaction.wrappers.engine as engine
import chain_reaction.backends.python.minimax_agent as min_agent
import matplotlib.pyplot as plt
import numpy as np

iterartions = []
gameiterations = []

# --------- UTILITY FUNCTIONS -----------
def uct_score(node, c_param) -> int:
    """ Calculates UCT score of a node """
    exploit = node.qscore / node.visits
    explore = math.sqrt(math.log(node.parent.visits, 10) / node.visits)
    return exploit + c_param * explore


def forward_roll_once(state, player) -> tuple:
    """
    Choose one action randomly and return next state tuple
    If the action leads to game over, return None and winner
    """

    # rollout policy: random
    valid_moves = engine.valid_board_moves(state, player)
    chosen_move = random.choice(valid_moves)

    # interact with env
    next_state = state[:]
    game_over = engine.interact_inplace(next_state, chosen_move, player)

    if game_over:
        return (None, player)
    else:
        return (next_state, 1 - player)


class MCTSVisitedNode:
    def __init__(self, state, parent, index, player):
        """
        Visited MCTS Node Class
        """

        self.state = state
        self.index = index
        self.player = player

        self.parent = parent
        self.children = []  # will be populated later

        self.unvisited = engine.valid_board_moves(state, player) if state else []
        self.is_terminal = False if state else True

        self.visits = 0
        self.qscore = 0

    def is_fully_expanded(self):
        return len(self.unvisited) == 0

    def expand(self):
        """ Construct child node from an untried action """

        # select one action
        action = self.unvisited.pop()

        # perform action and get state and game over
        next_state = self.state[:]
        game_over = engine.interact_inplace(next_state, action, self.player)
        next_state = None if game_over else next_state

        # construct child node and add to children
        child = MCTSVisitedNode(next_state, self, action, 1 - self.player)
        self.children.append(child)

        return child

    def best_child(self, c_param):
        """
        Choose child of node with maximum UCT score
        Return None if there are no children
        """
        b_score = -math.inf
        b_child = None
        c=0
        for child in self.children:
            score = uct_score(child, c_param)
            if score > b_score:
                b_score = score
                b_child = child
                c=c+1
        # print("the number of children are ",c)
        # print("ucb score of best child is = ", b_score)
        return b_child

    def simulate(self):
        """
        Play Random Games from Node
        Returns winner of game
        """

        state = self.state
        player = self.player
        count=0;
        # rollout till game over
        while state is not None:
            state, player = forward_roll_once(state, player)
            count=count+1
        
        # print("number of iterations for simulation required", count)
        # return winner for backpropagation
        return player

    def backpropagate(self, reward):
        """
        Update properties of node from reward
        Backpropagate from node to all ancestors
        """
        node = self

        while node is not None:
            node.visits += 1
            node.qscore += 1 if node.player == reward else -1
            node = node.parent


class MCTSRootNode(MCTSVisitedNode):
    def __init__(self, state, player):
        super().__init__(state, None, None, player)

    def best_action(self):
        """ Best move using score for exploitation only """
        p = self.best_child(c_param=0.0)
        print ("index of the node is = ", p.index)
        print("qscore of the node is = ", p.qscore)
        return p.index

    def tree_policy(self, c_param):
        """ Select a node to run simulation on """
        node = self

        # while node is fully_visited
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child(c_param)

        # get unexplored node
        return node


# ------------- OUTER FUNCTION --------------------
def best_action(board: list, player, time_limit, c_param, score0, score1) -> int:
    # setup
    global iterations
    global gameiterations
    time_start = time.perf_counter()
    rootnode = MCTSRootNode(board, player)
    # keeping the count of number of iterations
    count=0
    # time limited search
    while time.perf_counter() - time_start < time_limit:
        leafnode = rootnode.tree_policy(c_param)
        reward = leafnode.simulate()
        leafnode.backpropagate(reward)
        count=count+1

    data = np.full((5, 5), -10, dtype=float)
    prop_uct_list = [(i.index, uct_score(i, 0)) for i in rootnode.children]
    for i in range(len(rootnode.children)):
        row = int(prop_uct_list[i][0]/5)
        col = prop_uct_list[i][0]%5
        data[row][col]=prop_uct_list[i][1]
    # prop_list = [i.index for i in rootnode.children]
    print("number of children of the rootnode are", len(rootnode.children), prop_uct_list)
    print("number of iterations done by mcts = ",count)
    iterartions.append(count)
    gameiterations.append(len(iterartions))
    plt.clf()
    plt.plot(gameiterations, iterartions)
    plt.xlabel('game iterations label')
    plt.ylabel('mcts iterations label')
    plt.title('mcts iterations vs game iterations Chart Example')
    plt.savefig('mcts iterations.png')
    print("Player is", player)

    s0= min_agent.board_score(board, player)
    s1= min_agent.board_score(board, 1-player)
    print("score of the player is = ", s0)
    print("score of the other player is = ", s1)
    # print(board)

    score0.append(s0)
    score1.append(s1)
    if min_agent.board_score(board, player)>=20:
        plt.clf()
        plt.imshow(data, cmap='autumn') 
        plt.title( "2-D Heat Map" )
        plt.savefig('heatmap_ucb.png')
        done=False
        plt.show()
    
    print()
    return rootnode.best_action()
