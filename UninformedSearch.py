

# import pdb, for debugging, not used in final version

import itertools
import operator as op
import math
import random
import copy
from collections import deque

############################################################
#  N-Queens
############################################################


def ncr(n, r):

    # Helper method for calculating combinatorics
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = reduce(op.mul, xrange(n, n - r, -1))
    denom = reduce(op.mul, xrange(1, r + 1))
    return numer // denom


def num_placements_all(n):

    # For n identical queens, we have n^2 possible squares. So all possible
    # placements is actually equal to (n^2 C n). Scary.

    return ncr(n * n, n)


def num_placements_one_per_row(n):

    # We have n queens, which can each have n positions in their respective
    # row. That is (n^n) placements.

    return (n**n)


def n_queens_valid(board):
    # print_the_queens(board)

    # Firt I used the combinations function from itertools. It returns r
    # length tuples (r=2 in this case) of all possible orderings, no repeated elements.
    # With enumerate function you can get (row,col) for every Queen.
    # Thus combining these two functions gives us row and col of every Queen
    # and then all of the combinations of any 2 queens. The time
    # complexity of this function should be O(n^2).

    for a, b in itertools.combinations(enumerate(board), 2):

        # Two queens cannot be same row, same column

        if a[0] == b[0] or a[1] == b[1]:
            return False

        # Two queens cannot be same diagonal

        elif a[0] + a[1] == b[0] + b[1] or a[0] - a[1] == b[0] - b[1]:
            return False

    return True


def is_queen_safe(desired_row, desired_col, board):

    # The function determines if a newly added queen to the list is safe. We
    # could call n_queens_valid as well, but that would compare all
    # permutations of queens, increasing time complexity.
    # This function runs faster if you need to compare a single queen to a board of queens of size n, s.t.
    # n_queens_valid(board) returns ture.

    for row, col in enumerate(board):

        # Two queens cannot be same row, same column

        if row == desired_row or col == desired_col:
            return False

        # Two queens cannot be same diagonal

        elif row + col == desired_row + desired_col or row - col == desired_row - desired_col:
            return False

    return True


def print_the_queens(board):

    # Function for testing purposes. It prints the positions of the queens. I used it to visually
    # observe if the queens were added properly, that no queen was attacking any
    # other queen.

    result = []

    for row, col in enumerate(board):

        result.append(
            "There is a queen on row: {} col: {} on this board".format(row, col))
    print ("\n".join(result))


def print_the_board(board):

    # Function for testing purposes. It shows the board for every solution in a readable way. I used it to
    # visually analize solutions.

    size = len(board)
    for row in range(size):
        cells = (
            "Q" if board[row] == col else "-"
            for col in range(size)
        )
        print("".join(cells))


def n_queens_solution_generator(n):

    # Algorithm: DFS with backtracking to minimize number of searches.

    # Initialize the board with one Queen at row (0,0)

    board = [0]

    # We try all columns for second queen, starting from current_column = 0.

    current_column = 0
    solutions = []

    # When there are no more valid columns to try in the first row AND we need to
    # backtrack, there are no more solutions left. So we break out of this while
    # loop.

    while not current_column > n:

        found_position = False

        for col in range(current_column, n):

            if(is_queen_safe(len(board), col, board)):

                # if queen is safe, add it to the position. And reset
                # current_column to 0 for next row.

                board.append(col)
                current_column = 0
                found_position = True

                # This break is very important. We don't want to add more than
                # one queen per loop. Delete this break and very subtle
                # problems happen in n_queens_solutions(6)

                break

        # If the position is not valid

        if(not found_position):

            # If stack is empty, that means there are no more solutions left.
            # Finish.

            if(len(board) == 0):
                break

            else:

                # Backtrack to the previous row. Start from column_position + 1. That's where we
                # restart our search.

                next_column = board[-1] + 1

                # pop the queen from list

                board = board[:-1]
                current_column = next_column

        if(len(board) == n):

             # If there is a queen on every row, we have a solution. Yield that
            # solution and then backtrack like nothing happened to find the next
            # solution.
            yield board

            # Backtracking to last row.

            next_column = board[-1] + 1
            board = board[:-1]
            current_column = next_column


def n_queens_solutions(n):

    gen = n_queens_solution_generator(n)

    return gen

"""
TEST CASES USED TO DEBUG
solutions = n_queens_solutions(4)

print(next(solutions))
print(next(solutions))
print((list(n_queens_solutions(6))))
print((list(n_queens_solutions(9))))

print(len(list(n_queens_solutions(6))))
print(len(list(n_queens_solutions(9))))
"""
############################################################
# Lights Out
############################################################


class LightsOutPuzzle(object):

    def __init__(self, board):

        self.board = board
        self.row_length = len(board) - 1
        self.col_length = len(board[0]) - 1

    def get_board(self):
        return (self.board)

    def perform_move(self, row, col):

        # Instead of raising an error act like the move did not happen.
        if row >= self.row_length + 1 or col >= self.col_length + 1:
            return

        self.board[row][col] = not self.board[row][col]

        # check all four cases

        if row >= 1:
            self.board[row - 1][col] = not self.board[row - 1][col]
        if row <= self.row_length - 1:
            self.board[row + 1][col] = not self.board[row + 1][col]
        if col >= 1:
            self.board[row][col - 1] = not self.board[row][col - 1]
        if col <= self.col_length - 1:
            self.board[row][col + 1] = not self.board[row][col + 1]

    def scramble(self):

        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if random.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):
        # Creates a single list from list of lists and checks if any value is
        # True
        return not any(itertools.chain.from_iterable(self.board))

    def copy(self):
        return copy.deepcopy(self)

    def successors(self):

        # Generator comprehension. Applies every possible move and yields them.
        gen = (((row, col), self.apply_move(row, col)) for row in
               range(len(self.board)) for col in range(len(self.board[0])))
        return gen

    def apply_move(self, row, col):

        # Helper function to apply a single move and record it in a new board.
        new_board = self.copy()
        new_board.perform_move(row, col)
        return new_board

    def find_solution(self):

        # Explored board states will be stored here.
        explored = set()

        # deque is used. FIFO for BFS.

        q = deque()
        q.append(self)
        # print("This is the original board: starting state")
        # print_lists(self.get_board())

        # helper variables to bactrack the solution
        parent = {}
        parent[self] = None

        moves = {}
        moves[self] = None
        solution = []

        while not(len(q) == 0):

            board = q.popleft()

            # print(
            # "The board we are currently investigating and going over through their successors")
            # print_lists(board.get_board())

            explored.add(board.ListToTuple())

            for move, newBoard in board.successors():

                board_tuple = newBoard.ListToTuple()

                if board_tuple in explored:
                    continue
                else:
                    moves[newBoard] = move
                    parent[newBoard] = board
                if newBoard.is_solved():
                    node = board
                    while not parent[node] == None:
                        solution.append(tuple(moves[node]))
                        node = parent[node]
                    #
                    # BURAYI SAKIN UNUTMA
                    solution = [move] + solution
                    return list(reversed(solution))

                # add new node to deque only after testing
                q.append(newBoard)
                # print(
                #   "The following successors of the board are added to the queue for further investigation")
                # print_lists(newBoard.get_board())
        return None

    def ListToTuple(self):
        return tuple(tuple(row) for row in self.get_board())


def create_puzzle(rows, cols):
    return LightsOutPuzzle([[False for col in range(cols)]
                            for row in range(rows)])


def print_lists(lst):
    # helper function to visualize board

    for sublst in lst:

        print sublst,
        print

""""
TEST CASES USED FOR DEBUGGING
p = create_puzzle(2, 3)
print_lists(p.get_board())

b = [[True, True], [True, True]]
p = LightsOutPuzzle(b)
print_lists(p.get_board())
p = create_puzzle(3, 3)
(p.perform_move(1, 1))
print_lists(p.get_board())
p = create_puzzle(3, 3)
(p.perform_move(0, 0))

print_lists(p.get_board())
p = create_puzzle(2, 2)
for move, new_p in p.successors():
    print move, new_p.get_board()
for i in range(2, 6):
    p = create_puzzle(i, i + 1)
    print len(list(p.successors()))
p = create_puzzle(2, 3)
for row in range(2):
    for col in range(3):
        p.perform_move(row, col)

print(p.find_solution())

b = [[False, False, False], [False, False, False]]
b[0][0] = True
p = LightsOutPuzzle(b)
print(p.find_solution())
L4 = [[True, False, False, True], [True, False, True, True],
      [False, False, True, False], [True, False, False, True]]
p = LightsOutPuzzle(L4)
print(p.find_solution())

L6 = [[False, True, True, True, False], [True, False, True, False, True],
      [True, True, False, True, True], [True, False, True, False, True], [False, True, True, True, False]]
p = LightsOutPuzzle(L6)
print(p.find_solution())


p = create_puzzle(4, 4)
p.scramble()
print(p.find_solution())

p = create_puzzle(1, 1)
p.scramble()
print(p.find_solution())
"""
# Linear Disk Movement
############################################################

# I created a LinearDiskSolver Class. In the end, it looked very similar
# to the previous problem. But it was a great learning experience to write
# everything myself.


class LinearDiskSolver(object):

    def __init__(self, disks, length, n):

        self.board = list(disks)
        self.length = length
        self.n = n

    def get_board(self):

        return self.board

    def __str__(self):

        # Function is used for testing purposes.
        return ", ".join(str(x) for x in self.get_board())

    def copy(self):

        return copy.deepcopy(self)

    def is_solved_distinct(self):

        for i in xrange(self.length - self.n):

            if self.board[i] is not None:
                return False

        for i in xrange(self.length - self.n, self.length):

            if self.board[i] is not self.length - i:
                return False
        return True

    def is_solved_identical(self):

        for i in range(self.length - self.n):
            if self.board[i] != None:
                return False
        return True

    def apply_move(self, i, steps):

        # Helper function that modifies the board for all of the different
        # steps
        if (i + steps) < 0 or (i + steps) >= self.length:
            return
        self.board[i + steps] = self.board[i]
        self.board[i] = None

    def successors(self):

        # Similar approach taken from the LightsOutPuzzle problem
        board = self.get_board()

        length = self.length

        for i, cell in enumerate(board):

            if cell is not None:

                if i + 1 < length and board[i + 1] is None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, 1)
                    yield(tuple((i, i + 1)), newBoard)

                if i + 2 < length and board[i + 2] is None and board[i + 1] is not None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, 2)
                    yield(tuple((i, i + 2)), newBoard)

                if i - 1 >= 0 and board[i - 1] is None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, -1)
                    yield(tuple((i, i - 1)), newBoard)

                if i - 2 >= 0 and board[i - 2] is None and board[i - 1] is not None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, -2)
                    yield(tuple((i, i - 2)), newBoard)

    def solution_for_identical(self):

        # Returns a list of moves for identical disks problem
     # Explored board states will be stored here.
        explored = set()

    # deque is used. FIFO for BFS.

        q = deque()
        q.append(self)

    # helper variables to bactrack the solution
        parent = {}
        parent[self] = None

        moves = {}
        moves[self] = None
        solution = []

        while not(len(q) == 0):

            board = q.popleft()
            explored.add(tuple(board.get_board()))

            for move, newBoard in board.successors():

                boardTuple = tuple(newBoard.get_board())

                if boardTuple in explored:
                    continue

                else:
                    moves[newBoard] = move
                    parent[newBoard] = board

                if newBoard.is_solved_identical():

                    node = board

                    while not parent[node] == None:
                        solution.append(tuple(moves[node]))
                        node = parent[node]
                    solution = [move] + solution
                    return list(reversed(solution))
                q.append(newBoard)

        return None

    def solution_for_distinct(self):

        # Returns a list of moves for distinct disks problem

        explored = set()
        q = deque()
        q.append(self)

        #print("This is the original board: starting state")
       # print(self.get_board())

    # helper variables to bactrack the solution
        parent = {}
        parent[self] = None

        moves = {}
        moves[self] = None
        solution = []

        while not(len(q) == 0):

            board = q.popleft()

            # print(
            #    "The board we are currently investigating and going over through their successors")
            # print(board)

            explored.add(tuple(board.get_board()))

            for move, newBoard in board.successors():

                board_tuple = tuple(newBoard.get_board())
                if board_tuple in explored:
                    continue
                else:
                    moves[newBoard] = move
                    parent[newBoard] = board

                    # print(
                    #    "The following successors of the board are added to the queue for further investigation")
                   # print(newBoard)

                if newBoard.is_solved_distinct():
                    node = board
                    while not parent[node] == None:
                        solution.append(tuple(moves[node]))
                        node = parent[node]
                    solution = [move] + solution
                    return list(reversed(solution))
                q.append(newBoard)
        return None


def solve_identical_disks(length, n):

    # When l = 5, n = 2: disks[1,1,None,None,None]

    disks = tuple(1 if i < n else None for i in xrange(length))
    return LinearDiskSolver(disks, length, n).solution_for_identical()


def solve_distinct_disks(length, n):

    # When l = 5, n = 3: disks[1,2,3,None,None]

    disks = tuple(i + 1 if i < n else None for i in xrange(length))

    return LinearDiskSolver(disks, length, n).solution_for_distinct()


def create_disk_list(disks, length, n):
    # Function was for testing purposes.
    return LinearDiskSolver(disks, length, n)

"""
TESTING & DEBUGGING FOR Linear Disks 

disks = [None, None, 1, 1, 1, None]
p = create_disk_list(disks, 6, 3)
print(p.get_board())
gen = p.successors()
for a, b in list(gen):
    print a, b


print(solve_identical_disks(4, 2))
print(solve_identical_disks(4, 3))

print(solve_identical_disks(5, 2))
print(solve_identical_disks(5, 3))

print(solve_identical_disks(8, 6))


print(solve_distinct_disks(4, 2))
print(solve_distinct_disks(5, 2))
print(solve_distinct_disks(4, 3))
print(solve_distinct_disks(5, 3))

print(solve_distinct_disks(5, 3))
"""
