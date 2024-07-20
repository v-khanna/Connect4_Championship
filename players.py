import random
import time
import pygame
import math
from connect4 import connect4
import numpy as np


class connect4Player(object):
    def __init__(self, position, seed=0, CVDMode=False):
        self.position = position
        self.opponent = None
        self.seed = seed
        random.seed(seed)
        if CVDMode:
            global P1COLOR
            global P2COLOR
            P1COLOR = (227, 60, 239)
            P2COLOR = (0, 255, 0)

    def play(self, env: connect4, move: list) -> None:
        move = [-1]


class human(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        move[:] = [int(input("Select next move: "))]
        while True:
            if (
                int(move[0]) >= 0
                and int(move[0]) <= 6
                and env.topPosition[int(move[0])] >= 0
            ):
                break
            move[:] = [int(input("Index invalid. Select next move: "))]


class human2(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if self.position == 1:
                        pygame.draw.circle(
                            screen, P1COLOR, (posx, int(SQUARESIZE / 2)), RADIUS
                        )
                    else:
                        pygame.draw.circle(
                            screen, P2COLOR, (posx, int(SQUARESIZE / 2)), RADIUS
                        )
                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    move[:] = [col]
                    done = True


class randomAI(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)
        move[:] = [random.choice(indices)]


class stupidAI(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p:
                indices.append(i)
        if 3 in indices:
            move[:] = [3]
        elif 2 in indices:
            move[:] = [2]
        elif 1 in indices:
            move[:] = [1]
        elif 5 in indices:
            move[:] = [5]
        elif 6 in indices:
            move[:] = [6]
        else:
            move[:] = [0]


class minimaxAI(connect4Player):

    def __init__(self, position, depth=4, seed=1, CVDMode=False):
        super().__init__(position, seed, CVDMode)
        self.depth = depth

    def minimax(self, board, depth, maximizingPlayer):
        if depth == 0 or self.game_over(board):
            return self.evaluate_board(board, self.position)
        if maximizingPlayer:
            maxEval = float("-inf")
            for move in self.get_possible_moves(board):
                # Simulate the move
                temp_board = self.simulate_move(board, move, self.position)
                eval = self.minimax(temp_board, depth - 1, False)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            minEval = float("inf")
            for move in self.get_possible_moves(board):
                # Simulate the move
                temp_board = self.simulate_move(
                    board, move, 3 - self.position
                )  # Assuming 1 and 2 are player IDs
                eval = self.minimax(temp_board, depth - 1, True)
                minEval = min(minEval, eval)
            return minEval

    def game_over(self, board):
        # Check for win
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                if (
                    board[row][col]
                    == board[row][col + 1]
                    == board[row][col + 2]
                    == board[row][col + 3]
                    != 0
                ):
                    return True
        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT):
                if (
                    board[row][col]
                    == board[row + 1][col]
                    == board[row + 2][col]
                    == board[row + 3][col]
                    != 0
                ):
                    return True
        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT - 3):
                if (
                    board[row][col]
                    == board[row + 1][col + 1]
                    == board[row + 2][col + 2]
                    == board[row + 3][col + 3]
                    != 0
                ):
                    return True
        for row in range(3, ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                if (
                    board[row][col]
                    == board[row - 1][col + 1]
                    == board[row - 2][col + 2]
                    == board[row - 3][col + 3]
                    != 0
                ):
                    return True
        # Check for full board
        if all(board[0][col] != 0 for col in range(COLUMN_COUNT)):
            return True
        return False

    def get_possible_moves(self, board):
        return [col for col in range(COLUMN_COUNT) if board[0][col] == 0]

    def simulate_move(self, board, move, player):
        temp_board = [row[:] for row in board]  # Deep copy of the board
        for row in range(ROW_COUNT - 1, -1, -1):  # Start from the bottom row
            if temp_board[row][move] == 0:
                temp_board[row][move] = player
                break
        return temp_board

    def play(self, env: connect4, move: list) -> None:
        best_score = float("-inf")
        best_move = None
        for possible_move in self.get_possible_moves(env.getBoard()):
            # Simulate the move in a copy of the board
            temp_board = env.simulate_move(
                possible_move, self.position
            )  # Assuming this method exists
            score = self.minimax(temp_board, self.depth, False)
            if score > best_score:
                best_score = score
                best_move = possible_move
        move[0] = best_move

    def multiple_threats(self, board, player):
        threat_count = 0
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                window = [board[row][col + i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    threat_count += 1
        return threat_count

    def can_block_opponent_win_next(self, board, player):
        opponent = 3 - player
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                window = [board[row][col + i] for i in range(4)]
                if window.count(opponent) == 3 and window.count(0) == 1:
                    return True
        return False

    def calculate_mobility(self, board, player):
        mobility = 0
        for col in range(COLUMN_COUNT):
            if board[ROW_COUNT - 1][col] == 0:  # If the top row is empty in that column
                mobility += 1
        return mobility

    def evaluate_window(self, window, player):
        score = 0
        opponent = 1 if player == 2 else 2
        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2
        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 4
        return score

    def evaluate_board(self, board, player):
        board = np.array(board) if not isinstance(board, np.ndarray) else board
        score = 0
        opponent = 1 if player == 2 else 2
        rows, columns = board.shape
        # Center column preference
        center_array = [int(i) for i in list(board[:, columns // 2])]
        center_count = center_array.count(player)
        score += center_count * 3
        # Score Horizontal
        for row in range(rows):
            row_array = [int(i) for i in list(board[row, :])]
            for col in range(columns - 3):
                window = row_array[col : col + 4]
                score += self.evaluate_window(window, player)
        # Score Vertical
        for col in range(columns):
            col_array = [int(i) for i in list(board[:, col])]
            for row in range(rows - 3):
                window = col_array[row : row + 4]
                score += self.evaluate_window(window, player)
        # Score positive diagonal
        for row in range(rows - 3):
            for col in range(columns - 3):
                window = [board[row + i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player)
        # Score negative diagonal
        for row in range(rows - 3):
            for col in range(columns - 3):
                window = [board[row + 3 - i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player)

        score += self.multiple_threats(board, player) * 50
        if self.can_block_opponent_win_next(board, player):
            score += 30
        score += self.calculate_mobility(board, player) * 2

        return score


class alphaBetaAI(connect4Player):
    def __init__(self, position, depth=10, seed=0, CVDMode=False):
        super().__init__(position, seed, CVDMode)
        self.depth = depth

    def alpha_beta_pruning(self, board, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or self.game_over(board):
            return self.evaluate_board(board, self.position)

        if maximizingPlayer:
            maxEval = float("-inf")
            for move in self.get_possible_moves(board):
                temp_board = self.simulate_move(board, move, self.position)
                eval = self.alpha_beta_pruning(
                    temp_board, depth - 1, alpha, beta, False
                )
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = float("inf")
            for move in self.get_possible_moves(board):
                temp_board = self.simulate_move(
                    board, move, 3 - self.position
                )  # Assuming player IDs are 1 and 2
                eval = self.alpha_beta_pruning(temp_board, depth - 1, alpha, beta, True)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval

    def play(self, env: connect4, move: list) -> None:
        best_score = float("-inf")
        best_move = None
        for possible_move in self.get_possible_moves(env.getBoard()):
            temp_board = env.simulate_move(possible_move, self.position)
            score = self.alpha_beta_pruning(
                temp_board, self.depth, float("-inf"), float("inf"), True
            )
            if score > best_score:
                best_score = score
                best_move = possible_move
        move[0] = best_move

    def game_over(self, board):
        # Implement game over check
        # Check for win
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                if (
                    board[row][col]
                    == board[row][col + 1]
                    == board[row][col + 2]
                    == board[row][col + 3]
                    != 0
                ):
                    return True
        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT):
                if (
                    board[row][col]
                    == board[row + 1][col]
                    == board[row + 2][col]
                    == board[row + 3][col]
                    != 0
                ):
                    return True
        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT - 3):
                if (
                    board[row][col]
                    == board[row + 1][col + 1]
                    == board[row + 2][col + 2]
                    == board[row + 3][col + 3]
                    != 0
                ):
                    return True
        for row in range(3, ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                if (
                    board[row][col]
                    == board[row - 1][col + 1]
                    == board[row - 2][col + 2]
                    == board[row - 3][col + 3]
                    != 0
                ):
                    return True
        # Check for full board
        if all(board[0][col] != 0 for col in range(COLUMN_COUNT)):
            return True
        return False

    def get_possible_moves(self, board):
        # Implement logic to get possible moves
        return [col for col in range(COLUMN_COUNT) if board[0][col] == 0]

    def simulate_move(self, board, move, player):
        # Implement move simulation
        temp_board = [row[:] for row in board]
        for row in range(ROW_COUNT - 1, -1, -1):
            if temp_board[row][move] == 0:
                temp_board[row][move] = player
                break
        return temp_board

    def multiple_threats(self, board, player):
        threat_count = 0
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                window = [board[row][col + i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    threat_count += 1
        return threat_count

    def can_block_opponent_win_next(self, board, player):
        opponent = 3 - player
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                window = [board[row][col + i] for i in range(4)]
                if window.count(opponent) == 3 and window.count(0) == 1:
                    return True
        return False

    def calculate_mobility(self, board, player):
        mobility = 0
        for col in range(COLUMN_COUNT):
            if board[ROW_COUNT - 1][col] == 0:
                mobility += 1
        return mobility

    def evaluate_window(self, window, player):
        score = 0
        opponent = 1 if player == 2 else 2
        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2
        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 4
        return score

    def evaluate_board(self, board, player):
        board = np.array(board) if not isinstance(board, np.ndarray) else board
        score = 0
        opponent = 1 if player == 2 else 2
        rows, columns = board.shape
        # Center column preference
        center_array = [int(i) for i in list(board[:, columns // 2])]
        center_count = center_array.count(player)
        score += center_count * 3
        # Score Horizontal
        for row in range(rows):
            row_array = [int(i) for i in list(board[row, :])]
            for col in range(columns - 3):
                window = row_array[col : col + 4]
                score += self.evaluate_window(window, player)
        # Score Vertical
        for col in range(columns):
            col_array = [int(i) for i in list(board[:, col])]
            for row in range(rows - 3):
                window = col_array[row : row + 4]
                score += self.evaluate_window(window, player)
        # Score positive diagonal
        for row in range(rows - 3):
            for col in range(columns - 3):
                window = [board[row + i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player)
        # Score negative diagonal
        for row in range(rows - 3):
            for col in range(columns - 3):
                window = [board[row + 3 - i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player)

        score += self.multiple_threats(board, player) * 50
        if self.can_block_opponent_win_next(board, player):
            score += 30
        score += self.calculate_mobility(board, player) * 2

        return score


SQUARESIZE = 100
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
P1COLOR = (255, 0, 0)
P2COLOR = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
