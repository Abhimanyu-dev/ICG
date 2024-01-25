import time
import csv
import math
from random import randint
from re import T
from tempfile import tempdir
import copy
from threading import activeCount

X = "X"
O = "O"
EMPTY = None

def player(board):
    """
    Returns player who has the next turn on a board.
    """

    if terminal(board):
        return 0

    counter = 0

    for row in board:
        for cell in row:
            if cell is not None:
                counter += 1

    if counter % 2 == 0:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    action_array = []
    if terminal(board):
        return 0
    else:
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    action_array.append((i, j))
        return action_array

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    board_copy = copy.deepcopy(board)
    chance = player(board)
    if terminal(board):
        return 0

    if board[action[0]][action[1]] is not None: 
        raise Exception("Invalid Action")
    
    if action[0]>2 or action[1]>2:
        raise Exception("Invalid Action")

    board_copy[action[0]][action[1]] = chance
    
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    rows = copy.deepcopy(board)
    diagonals = [[],[]]
    columns = []
    # print(rows)

    for i in range(3):
        column = []
        for j in range(3):
            cell = rows[j][i]
            column.append(cell)
        columns.append(column)

    # print(rows)
    for i in range(3):
        diagonals[0].append(rows[i][i])
        diagonals[1].append(rows[i][2-i])

    # print(diagonals)
    winner = None

    for row in rows:
        if row[0] == row[1] == row[2] and row[0] is not None:
            winner = row[0]
    for row in columns:
        if row[0] == row[1] == row[2] and row[0] is not None:
            winner = row[0]
    for row in diagonals:
        if row[0] == row[1] == row[2] and row[0] is not None :
            winner = row[0]

    return winner


    



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    check = False
    if winner(board) is not None:
        check = True
    else:
        check = True
        for row in board:
            for cell in row:
                if cell is None:
                    check = False
    
    return check


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) is None:
            return 0
        elif winner(board) == X:
            return 1
        elif winner(board) == O:
            return -1
    
    return 0



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    chance = player(board)
    optimal_action = ()

    if terminal(board):
        return 0
    
    if empty(board):
        return actions(board)[randint(0,7)]
    
    if chance == X:
        v = -99999

        for action in actions(board):
            # print (result(board, action))
            value = MinValue(board, action)
            if (value>v):
                v = value
                optimal_action = action
    else:
        v = 999999
        for action in actions(board):
            # print (result(board, action))
            value = MaxValue(board, action)
            if (value<v):
                v = value
                optimal_action = action

    return optimal_action


def empty(board):
    check = True
    for row in board:
        for cell in row:
            if cell is not None:
                check = False
    return check


def MaxValue(board, action):
    v = -999999
    if action != 0:
        board = result(board, action)
    
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = max(v, MinValue(board, action))
    return v

def MinValue(board, action):
    v = 9999999
    if action != 0:
        board = result(board, action)
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = min(v, MaxValue(board, action))
    return v

def convert(board):
    new_board = []
    mapper = [X, O, EMPTY]
    for i in range(3):
        row = []
        for j in range(3):
            row.append(mapper[int(board[3*i+j])])
        new_board.append(row)
    return new_board

def display(board):
    for row in board:
        print(row)


winner_mapper = {
    X: "1",
    O: "0",
    None: "2"
}

solution_file = open("solution(1).csv","w", newline="")
solution_writer = csv.writer(solution_file)
solution_writer.writerow([
    "ID",
    "POS_1",
    "POS_2",
    "POS_3",
    "POS_4",
    "POS_5",
    "POS_6",
    "POS_7",
    "POS_8",
    "POS_9",
    "Decision"
])
with open("./solution_final.csv", "r") as file:
    reader = csv.reader(file)
    for index, lines in enumerate(reader):
        if index == 0:
            continue
        row = lines[:-1]
        board = convert(lines[1:-1])
        # display(board)
        while not terminal(board):
            action = minimax(board)
            board = result(board, action)
            # display(board)
        row.append(winner_mapper[winner(board)])
        solution_writer.writerow(row)
        # print(winner_mapper[winner(board)])
        # time.sleep(10)










