import pickle

import chess
import chess.pgn
import numpy as np
import csv

from games_to_matrices import (board_to_matrix, get_move,
                               split_into_matrices_and_values)

if __name__ == "__main__":
    path = "./data/lichess_db_puzzle.csv"
    max_moves = 500000
    five_percent = max_moves // 20
    matrices = {}
    count = 0

    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # Get the board from the FEN that is on the 1 column
            board = chess.Board(row[1])
            game = chess.pgn.Game()
            game.setup(board)
            # The sequence of UCI moves is on the 2 column
            moves = row[2].split(" ")
            # Make the first move
            move = moves.pop(0)
            move = board.parse_uci(move)
            board.push(move)
            game.add_main_variation(move)
            game = game.next()
            first = True
            while moves:
                if not first:
                    # Get the move
                    move_one_hot = get_move(game)
                    # Add the move to the dictionary
                    if move_one_hot not in matrices[matrix]:
                        matrices[matrix][move_one_hot] = 0
                    matrices[matrix][move_one_hot] += 1
                else:
                    first = False
                board = game.board()
                matrix = board_to_matrix(board)
                if matrix not in matrices:
                    matrices[matrix] = {}
                # Make the next move
                move = moves.pop(0)
                move = board.parse_uci(move)
                board.push(move)
                game.add_main_variation(move)
                game = game.next()
                count += 1
                if count % five_percent == 0:
                    print(f"Finished {count / max_moves * 100}% of puzzles")
            if count > max_moves:
                break

    print("Finished making matrices.")

    just_matrices, just_values = split_into_matrices_and_values(matrices)

    print("Finished converting matrices to numpy array and values to probabilities.")

    # Save just_matrices and just_values
    np.save("data/matrices_puzzles.npy", just_matrices)
    with open("data/values_puzzles.pkl", "wb") as f:
        pickle.dump(just_values, f)

    print("Finished saving matrices and values.")
