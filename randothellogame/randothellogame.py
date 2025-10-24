'''
randothellogame module

sets up a RandOthello game closely following the book's framework for games

RandOthelloState is a class that will handle our state representation, then we've 
got stand-alone functions for player, actions, result and terminal_test

Differing from the book's framework, is that utility is *not* a stand-alone 
function, as each player might have their own separate way of calculating utility


'''
#The advanced search algorithm was done with the assistance of AI(ChatGPT), the extend is minimal ~ moderate. 
#The use of AI has been labeled.

# run: python randothellogame.py
import math
import random
import copy

WHITE = 1
BLACK = -1
EMPTY = 0
BLOCKED = -2
SIZE = 8
SKIP = "SKIP"

class OthelloPlayerTemplate:
    '''Template class for an Othello Player

    An othello player *must* implement the following methods:

    get_color(self) - correctly returns the agent's color

    make_move(self, state) - given the state, returns an action that is the agent's move
    '''
    def __init__(self, mycolor):
        self.color = mycolor

    def get_color(self):
        return self.color

    def make_move(self, state):
        '''Given the state, returns a legal action for the agent to take in the state
        '''
        return None

class HumanPlayer(OthelloPlayerTemplate):
    def __init__(self, mycolor):
        self.color = mycolor

    def get_color(self):
        return self.color

    def make_move(self, state):
        curr_move = None
        legals = actions(state)
        while curr_move == None:
            display(state)
            if self.color == 1:
                print("White ", end='')
            else:
                print("Black ", end='')
            print(" to play.")
            print("Legal moves are " + str(legals))
            move = input("Enter your move as a r,c pair:")
            if move == "":
                return legals[0]

            if move == SKIP and SKIP in legals:
                return move

            try:
                movetup = int(move.split(',')[0]), int(move.split(',')[1])
            except:
                movetup = None
            if movetup in legals:
                curr_move = movetup
            else:
                print("That doesn't look like a legal action to me")
        return curr_move

class RandomPlayer(OthelloPlayerTemplate):
    def __init__(self, mycolor):
        self.color = mycolor

    def get_color(self):
        return self.color

    def make_move(self, state):
        legals = actions(state)
        if not legals:
            return SKIP
        curr_move = random.choice(legals) 

        return curr_move
    

class MinimaxPlayer(OthelloPlayerTemplate):
    def __init__(self, mycolor, max_depth=3):
        self.color = mycolor
        self.max_depth = max_depth

    def get_color(self):
        return self.color

    def make_move(self, state):
        value, move = self._max_value(state, 0)
        if move is None:
            legals = actions(state)
            move = random.choice(legals)
        return move

    def _max_value(self, state, depth):
        if self._is_cutoff(state, depth):
            return self._utility(state), None
        v, move = float('-inf'), None
        for a in actions(state):
            s2 = result(state, a)
            if s2 is None:    
                continue
            v2, a2 = self._min_value(s2, depth + 1)
            if v2 > v:
                v, move = v2, a
        return v, move

    def _min_value(self, state, depth):
        if self._is_cutoff(state, depth):
            return self._utility(state), None
        v, move = float('inf'), None
        for a in actions(state):
            s2 = result(state, a)
            if s2 is None:
                continue
            v2, a2 = self._max_value(s2, depth + 1)
            if v2 < v:
                v, move = v2, a
        return v, move

    def _is_cutoff(self, state, depth):
        if terminal_test(state):
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _utility(self, state):
        score = 0
        for i in range(SIZE):
            for j in range(SIZE):
                if state.board_array[i][j] == self.color:
                    score += 1
                elif state.board_array[i][j] == -self.color:
                    score -= 1
        return score


class AlphabetaPlayer(OthelloPlayerTemplate):
    def __init__(self, mycolor, max_depth=3):
        self.color = mycolor
        self.max_depth = max_depth

    def get_color(self):
        return self.color

    def make_move(self, state):
        value, best_move = self._max_value(state, 0, float('-inf'), float('inf'))
        if best_move is None: 
            legals = actions(state)
            best_move = random.choice(legals)
        return best_move

    def _max_value(self, state, depth, alpha, beta):
        if self._is_cutoff(state, depth):
            return self._utility(state), None
        v, move = float('-inf'), None
        for a in actions(state):
            s2 = result(state, a)
            if s2 is None:
                continue
            v2, a2 = self._min_value(s2, depth + 1, alpha, beta)
            if v2 > v:
                v, move = v2, a
            alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    def _min_value(self, state, depth, alpha, beta):
        if self._is_cutoff(state, depth):
            return self._utility(state), None
        v, move = float('inf'), None
        for a in actions(state):
            s2 = result(state, a)
            if s2 is None:
                continue
            v2, a2 = self._max_value(s2, depth + 1, alpha, beta)
            if v2 < v:
                v, move = v2, a
            beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    def _is_cutoff(self, state, depth):
        if terminal_test(state):
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _utility(self, state):
        score = 0
        for i in range(SIZE):
            for j in range(SIZE):
                if state.board_array[i][j] == self.color:
                    score += 1
                elif state.board_array[i][j] == -self.color:
                    score -= 1
        return score


#preparations for the Monte Carlo Tree Search
class Node:
    def __init__(self, state, mycolor,parent=None):
        self.state = state
        self.parent = parent
        self.untried_actions = list(actions(state)) #AI assisted
        self.children = []
        self.visits = 0
        self.values = 0.0
        self.mycolor = mycolor
        self.action = None
        

    def Expand(self):
        if not self.untried_actions:
            return None
        action = self.untried_actions.pop()
        next_state = result(self.state, action)
        if next_state is None:
            return None
        child = Node(next_state, self.mycolor, parent=self)
        child.action = action
        self.children.append(child)
        return child
    
    def Select(self):
        if not self.children or self.untried_actions:
            return self
        best_ucb = float('-inf') #UCB1 formula was assisted with AI
        best_child = None
        parent_visits = max(1, self.visits)
        for child in self.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                ucb = (child.values/ child.visits) + math.sqrt(math.log(parent_visits) / child.visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child or self
    
    def Simulate(self):
        current_state = copy.deepcopy(self.state)
        while not terminal_test(current_state):
            possible_actions = actions(current_state)
            if not possible_actions: #possible actions were assisted with AI
                try:
                    next_state = result(current_state, SKIP)
                except Exception:
                    break
                if next_state is None:
                    break
                current_state = next_state
                continue
            action = random.choice(possible_actions)
            next_state = result(current_state, action)
            if next_state is None:
                break
            current_state = next_state

        mcount = 0
        ocount = 0
    
        for i in range(SIZE):
            for j in range(SIZE):
                if current_state.board_array[i][j] == self.mycolor:
                    mcount += 1
                elif current_state.board_array[i][j] == -self.mycolor:
                    ocount += 1
        return float(mcount - ocount)

    def Backpropagate(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.values += reward
            node = node.parent
    

class AdvancedPlayer(OthelloPlayerTemplate):
    def __init__(self, mycolor, max_depth=3):
        self.color = mycolor
        self.max_depth = max_depth
    
    def get_color(self):
        return self.color
    
    def make_move(self, state):

        m = self._monte_carlo_tree_search(state, iters=32)
        if m is not None:
            return m
        
        value, best_move = self._max_value(state, 0, float('-inf'), float('inf'))
        if best_move is None:
            legals = actions(state)
            best_move = random.choice(legals)
        return best_move
    
    def _max_value(self, state, depth, alpha, beta):
        if self._is_cutoff(state, depth):
            return self._utility(state), None
        v, move = float('-inf'), None
        for a in actions(state):
            s2 = result(state, a)
            if s2 is None:
                continue
            v2, a2 = self._min_value(s2, depth + 1, alpha, beta)
            if v2 > v:
                v, move = v2, a
            alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    def _min_value(self, state, depth, alpha, beta):
        if self._is_cutoff(state, depth):
            return self._utility(state), None
        v, move = float('inf'), None
        for a in actions(state):
            s2 = result(state, a)
            if s2 is None:
                continue
            v2, a2 = self._max_value(s2, depth + 1, alpha, beta)
            if v2 < v:
                v, move = v2, a
            beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    def _is_cutoff(self, state, depth):
        if terminal_test(state):
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _utility(self, state):
        score = 0
        for i in range(SIZE):
            for j in range(SIZE):
                if state.board_array[i][j] == self.color:
                    score += 1
                elif state.board_array[i][j] == -self.color:
                    score -= 1
        return score

    def _monte_carlo_tree_search(self, state, iters=32):
        tree = Node(state, self.color)

        def _best_child_by_visits(node): #assisted with AI
            if not node.children:
                return None
            return max(node.children, key=lambda ch: ch.visits)
        
        for _ in range(iters):
            leaf = tree.Select()
            child = leaf.Expand() or leaf
            reward = child.Simulate()
            child.Backpropagate(reward)

        best_child = _best_child_by_visits(tree)
        return best_child.action if best_child is not None else None




        

class RandOthelloState:
    '''A class to represent an othello game state'''

    def __init__(self, currentplayer, otherplayer, board_array = None, num_skips = 0):
        if board_array != None:
            self.board_array = board_array
        else:
            self.board_array = [[EMPTY] * SIZE for i in range(SIZE)]
            self.board_array[3][3] = WHITE
            self.board_array[4][4] = WHITE
            self.board_array[3][4] = BLACK
            self.board_array[4][3] = BLACK
            x1 = random.randrange(8)
            x2 = random.randrange(8)
            self.board_array[x1][0] = BLOCKED
            self.board_array[x2][7] = BLOCKED
        self.num_skips = num_skips
        self.current = currentplayer
        self.other = otherplayer


def player(state):
    return state.current

def actions(state):
    '''Return a list of possible actions given the current state
    '''
    legal_actions = []
    for i in range(SIZE):
        for j in range(SIZE):
            if result(state, (i,j)) != None:
                legal_actions.append((i,j))
    if len(legal_actions) == 0:
        legal_actions.append(SKIP)
    return legal_actions

def result(state, action):
    '''Returns the resulting state after taking the given action

    (This is the workhorse function for checking legal moves as well as making moves)

    If the given action is not legal, returns None

    '''
    # first, special case! an action of SKIP is allowed if the current agent has no legal moves
    # in this case, we just skip to the other player's turn but keep the same board
    if action == SKIP:
        newstate = RandOthelloState(state.other, state.current, copy.deepcopy(state.board_array), state.num_skips + 1)
        return newstate

    if state.board_array[action[0]][action[1]] != EMPTY:
        return None

    color = state.current.get_color()
    # create new state with players swapped and a copy of the current board
    newstate = RandOthelloState(state.other, state.current, copy.deepcopy(state.board_array))

    newstate.board_array[action[0]][action[1]] = color
    
    flipped = False
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for d in directions:
        i = 1
        count = 0
        while i <= SIZE:
            x = action[0] + i * d[0]
            y = action[1] + i * d[1]
            if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
                count = 0
                break
            elif newstate.board_array[x][y] == -1 * color:
                count += 1
            elif newstate.board_array[x][y] == color:
                break
            else:
                count = 0
                break
            i += 1

        if count > 0:
            flipped = True

        for i in range(count):
            x = action[0] + (i+1) * d[0]
            y = action[1] + (i+1) * d[1]
            newstate.board_array[x][y] = color

    if flipped:
        return newstate
    else:  
        # if no pieces are flipped, it's not a legal move
        return None

def terminal_test(state):
    '''Simple terminal test
    '''
    # if both players have skipped
    if state.num_skips == 2:
        return True

    # if there are no empty spaces
    empty_count = 0
    for i in range(SIZE):
        for j in range(SIZE):
            if state.board_array[i][j] == EMPTY:
                empty_count += 1
    if empty_count == 0:
        return True
    return False

def display(state):
    '''Displays the current state in the terminal window
    '''
    print('  ', end='')
    for i in range(SIZE):
        print(i,end='')
    print()
    for i in range(SIZE):
        print(i, '', end='')
        for j in range(SIZE):
            if state.board_array[j][i] == WHITE:
                print('W', end='')
            elif state.board_array[j][i] == BLACK:
                print('B', end='')
            elif state.board_array[j][i] == BLOCKED:
                print('X', end='')
            else:
                print('-', end='')
        print()

def display_final(state):
    '''Displays the score and declares a winner (or tie)
    '''
    wcount = 0
    bcount = 0
    for i in range(SIZE):
        for j in range(SIZE):
            if state.board_array[i][j] == WHITE:
                wcount += 1
            elif state.board_array[i][j] == BLACK:
                bcount += 1

    print("Black: " + str(bcount))
    print("White: " + str(wcount))
    if wcount > bcount:
        print("White wins")
    elif wcount < bcount:
        print("Black wins")
    else:
        print("Tie")

def play_game(p1 = None, p2 = None):
    '''Plays a game with two players. By default, uses two humans
    '''
    if p1 == None:
        p1 = HumanPlayer(BLACK)
    if p2 == None:
        p2 = HumanPlayer(WHITE)

    s = RandOthelloState(p1, p2)
    while True:
        action = p1.make_move(s)
        if action not in actions(s):
            print("Illegal move made by Black")
            print("White wins!")
            return
        s = result(s, action)
        if terminal_test(s):
            print("Game Over")
            display(s)
            display_final(s)
            return
        action = p2.make_move(s)
        if action not in actions(s):
            print("Illegal move made by White")
            print("Black wins!")
            return
        s = result(s, action)
        if terminal_test(s):
            print("Game Over")
            display(s)
            display_final(s)
            return

def main():
    # black = MinimaxPlayer(BLACK, max_depth=4)
    # black = RandomPlayer(WHITE)
    # black = AdvancedPlayer(BLACK, max_depth=4)
    black = AlphabetaPlayer(BLACK, max_depth=4)

    white = RandomPlayer(WHITE)
    # white = AlphabetaPlayer(WHITE)

    play_game(black, white)

    # play_game()

if __name__ == '__main__':
    main()
