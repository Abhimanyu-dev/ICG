# importing libraries
import csv
import random

# get a list of sums which a grid location is a part of
grid_loc_to_sum_indexes = {
    (0, 0): [0, 3, 7],
    (0, 1): [0, 4],
    (0, 2): [0, 5, 6],
    (1, 0): [1, 3],
    (1, 1): [1, 4, 6, 7],
    (1, 2): [1, 5],
    (2, 0): [2, 3, 6],
    (2, 1): [2, 4],
    (2, 2): [2, 5, 7]
}

# index for storing count of sums
index_of_sum_count = [4, 3, 2, 1, 0]

# convert 0,1,2 to 1,-1,0 format
def convert_to_new_format(grid):
    
    for x in range(9):
        if grid[x] == '0':
            grid[x] = 1
        elif grid[x] == '1':
            grid[x] = -1
        elif grid[x] == '2':
            grid[x] = 0

    return grid

def convert_to_old_format(grid):
    
    for x in range(9):
        if grid[x] == 1:
            grid[x] = '0'
        elif grid[x] == -1:
            grid[x] = '1'
        elif grid[x] == 0:
            grid[x] = '2'

    return grid


# convert 1D grid to a 2D grid
def convert_1D_to_2D_grid(grid):
    converted_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for x in range(9):
        converted_grid[x // 3][x % 3] = grid[x]

    return converted_grid


# calculate the 8 sums
def calc_8_sums(grid):
    grid_sums = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(3):
        for j in range(3):

            grid_sums[8] += grid[i][j]
            for x in grid_loc_to_sum_indexes[(i, j)]:
                grid_sums[x] += grid[i][j]

    return grid_sums


# compare function, returns:
# 1 : sum1 > sum2
# -1: sum1 < sum2
# 0 : sum1 = sum2
def compare(sum1, sum2, player):

    if player == 1:
        comparison_order = [0, -1, 1, -2, 2]
       
    elif player == -1:
        comparison_order = [-1, 0, -2, 1, -3]


    for x in comparison_order:
        if sum1[x] > sum2[x]:
                return 1
        elif sum1[x] < sum2[x]:
                return -1
    return 0


# check if given player has won
def check_win(grid_sums, player):
    
    for x in range(8):
        if grid_sums[x] == -3 * player:
            return player

    return 0


# check if grid is full
def check_grid_full(grid):
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:
                return 0

    return 1


# get list of sums for a given i, j
def get_sums(i, j, grid_sums):
    sum_indexes = grid_loc_to_sum_indexes[(i, j)]
    sum_list = [0, 0, 0, 0, 0]

    for x in sum_indexes:
        sum_list[index_of_sum_count[grid_sums[x] + 2]] += 1

    return sum_list

# return the best move
def next_best_move(grid, grid_sums, curr_player):
    
    best_move = (0, 0)
    best_sums = [0, 0, 0, 0, 0]

    for i in range(3):
        for j in range(3):

            if grid[i][j] == 0:
                curr_sums = get_sums(i, j, grid_sums)

                if compare(curr_sums, best_sums, curr_player) == 1:
                    best_sums = curr_sums
                    best_move = (i, j)

    return best_move


# predicting winner
def predict_winner(grid):
    grid = convert_to_new_format(grid)
    grid = convert_1D_to_2D_grid(grid)
    grid_sums = calc_8_sums(grid)

    # assings curr_player to 1 for X and -1 for O
    curr_player = 1 - 2*grid_sums[8]
    if curr_player != 1 and curr_player != -1:
        return 0
    while not(check_win(grid_sums, curr_player)):

        if check_grid_full(grid):
            return 0
        
        move = next_best_move(grid, grid_sums, curr_player)

        grid[move[0]][move[1]] = curr_player
        curr_player *= -1

        grid_sums = calc_8_sums(grid)

    return curr_player


def main():

    # grid = [int(x) for x in input("Enter elements separated by space: ").split()]
    # print(predict_winner(grid))

    # 0 cross
    # 1 zero
    # 2 blank
    i = 1
    mapper = {
        -1: "1",
        1: "0",
        0: "2"    
        }
    with open('labels.csv', 'r') as file:
        
        csv_reader = csv.reader(file)

        for row in csv_reader:
            winner = predict_winner(row)
            answer_file = open("solution.csv", "a", newline="")

            row = convert_to_old_format(row)
            row.append(mapper[winner])
            csv.writer(answer_file).writerow(row)
#    grid = [-1, 0, 0, 1, 0, 0, 1, 0, 0]
#    print(predict_winner(grid))

# Check if the script is being run directly
if __name__ == "__main__":
    main()