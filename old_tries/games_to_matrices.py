import pickle

import chess
import chess.pgn
import numpy as np


def get_move(game):
    # Get move in a format of (from_square_id, to_square_id). Example: (12, 28)
    uci_move = game.move.uci()
    move = (
        ord(uci_move[0]) - 97 + (int(uci_move[1]) - 1) * 8,
        ord(uci_move[2]) - 97 + (int(uci_move[3]) - 1) * 8,
    )
    # Make one-hot encoding of the move
    move_one_hot = np.zeros((2, 64), dtype=np.uint8)
    to_add = 1
    # If promotion, set to_add to the piece that is promoted to
    if len(uci_move) == 5:
        if uci_move[4] == "q":
            to_add = 5
        elif uci_move[4] == "r":
            to_add = 4
        elif uci_move[4] == "b":
            to_add = 3
        elif uci_move[4] == "n":
            to_add = 2
    move_one_hot[0, move[0]] = to_add
    move_one_hot[1, move[1]] = to_add
    # Convert move_one_hot to a tuple of tuples
    move_one_hot = tuple(tuple(i) for i in move_one_hot)
    return move_one_hot


def board_to_matrix(board):
    color = board.turn
    matrix = np.zeros((12, 8, 8), dtype=np.int8)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece:
                symbol = piece.symbol()
                multiplier = 1 if color == piece.color else -1
                if symbol == "P":
                    matrix[0, j, i] = multiplier
                elif symbol == "N":
                    matrix[1, j, i] = multiplier
                elif symbol == "B":
                    matrix[2, j, i] = multiplier
                elif symbol == "R":
                    matrix[3, j, i] = multiplier
                elif symbol == "Q":
                    matrix[4, j, i] = multiplier
                elif symbol == "K":
                    matrix[5, j, i] = multiplier
                    if board.has_kingside_castling_rights(chess.WHITE):
                        matrix[5, j, i] += multiplier
                    if board.has_queenside_castling_rights(chess.WHITE):
                        matrix[5, j, i] += 2 * multiplier
                elif symbol == "p":
                    matrix[6, j, i] = multiplier
                elif symbol == "n":
                    matrix[7, j, i] = multiplier
                elif symbol == "b":
                    matrix[8, j, i] = multiplier
                elif symbol == "r":
                    matrix[9, j, i] = multiplier
                elif symbol == "q":
                    matrix[10, j, i] = multiplier
                elif symbol == "k":
                    matrix[11, j, i] = multiplier
                    if board.has_kingside_castling_rights(chess.BLACK):
                        matrix[11, j, i] += multiplier
                    if board.has_queenside_castling_rights(chess.BLACK):
                        matrix[11, j, i] += 2 * multiplier
    # Make the matrix hashable
    matrix = matrix.tobytes()
    return matrix


def split_into_matrices_and_values(matrices):
    # Split matrices into array of keys and list of values
    just_matrices = np.array(
        [np.frombuffer(i, dtype=np.int8).reshape((12, 8, 8)) for i in matrices.keys()]
    )
    just_values = [list(mat.items()) for mat in matrices.values()]
    # Get probabilities of each move instead of counts
    for i, values in enumerate(just_values):
        total = sum([j[1] for j in values])
        just_values[i] = [(j[0], j[1] / total) for j in values]
    return just_matrices, just_values


def process_elos(elo, offset, total_games):
    print(f"Processing elos {elo[0]}-{elo[1]}")

    print("Ended taking this rated games. Found", len(offset), "games")

    matrices = process_offsets(offset, total_games, pgn)

    print("Finished making matrices.")

    just_matrices, just_values = split_into_matrices_and_values(matrices)

    print("Finished converting matrices to numpy array and values to probabilities.")

    # Save just_matrices and just_values
    np.save(f"data/matrices_{elo[1]}.npy", just_matrices)
    print("Finished saving matrices.")
    with open(f"data/values_{elo[1]}.pkl", "wb") as f:
        pickle.dump(just_values, f)

    print("Finished saving values.")


def get_offsets(pgn):
    offsets = {
        800: [],
        1200: [],
        1600: [],
        2000: [],
        2400: [],
        2800: [],
    }

    counts = {
        800: 0,
        1200: 0,
        1600: 0,
        2000: 0,
        2400: 0,
        2800: 0,
    }
    while True:
        if sum(counts.values()) > 7000 * 5 + 6000:
            break
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)
        if not headers:
            break
        elos = headers.get("WhiteElo"), headers.get("BlackElo")
        if any(not i.isnumeric() for i in elos):
            continue
        if int(elos[0]) <= 800:
            if counts[800] > 7000:
                continue
            if counts[800] % 1000 == 0:
                print(counts)
            offsets[800].append(offset)
            counts[800] += 1
        elif int(elos[0]) <= 1200:
            if counts[1200] > 7000:
                continue
            if counts[1200] % 1000 == 0:
                print(counts)
            offsets[1200].append(offset)
            counts[1200] += 1
        elif int(elos[0]) <= 1600:
            if counts[1600] > 6000:
                continue
            if counts[1600] % 1000 == 0:
                print(counts)
            offsets[1600].append(offset)
            counts[1600] += 1
        elif int(elos[0]) <= 2000:
            if counts[2000] > 7000:
                continue
            if counts[2000] % 1000 == 0:
                print(counts)
            offsets[2000].append(offset)
            counts[2000] += 1
        elif int(elos[0]) <= 2400:
            if counts[2400] > 7000:
                continue
            if counts[2400] % 1000 == 0:
                print(counts)
            offsets[2400].append(offset)
            counts[2400] += 1
        elif int(elos[0]) <= 2800:
            if counts[2800] > 7000:
                continue
            if counts[2800] % 1000 == 0:
                print(counts)
            offsets[2800].append(offset)
            counts[2800] += 1
    return offsets, counts


def process_offsets(offsets, total_games, pgn):
    matrices = {}

    count = 0
    five_percent = total_games // 20
    for offset in offsets:
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        if count % five_percent == 0:
            print(f"Finished {count / total_games * 100}% of this rated games")
        if count > total_games:
            break
        if game.errors:
            continue
        board = game.board()
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
        count += 1
    return matrices


if __name__ == "__main__":
    path = "./lichess_2022_02.pgn"
    pgn = open(path)
    offsets, counts = get_offsets(pgn)
    for i in range(400, 3201, 400):
        process_elos((i, i + 400), offsets[i + 400], counts[i + 400])
