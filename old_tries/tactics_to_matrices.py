import pickle

import chess
import chess.pgn
import numpy as np

from games_to_matrices import (board_to_matrix, get_move,
                               split_into_matrices_and_values)

if __name__ == "__main__":
    path = "./chess_tactics_pgn/tactics.pgn"

    pgn = open(path)
    print("Opened pgn file.")
    matrices = {}

    while True:
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break
        if headers.get("FEN") is None:
            continue
        game = chess.pgn.read_game(pgn)
        if game.errors:
            continue
        game = game.next()
        first = True
        while game:
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
            game = game.next()

    print("Finished making matrices.")

    just_matrices, just_values = split_into_matrices_and_values(matrices)

    print("Finished converting matrices to numpy array and values to probabilities.")

    # Save just_matrices and just_values
    np.save("data/matrices_tactics.npy", just_matrices)
    with open("data/values_tactics.pkl", "wb") as f:
        pickle.dump(just_values, f)

    print("Finished saving matrices and values.")
