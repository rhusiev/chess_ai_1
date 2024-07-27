"""Converts a pgn file to a dictionary of fens and probability of next fens occuring."""
import pickle

import chess
import chess.pgn


def process_elos(elo, offset, total_games):
    print(f"Processing elos {elo[0]}-{elo[1]}")

    print("Ended taking this rated games. Found", len(offset), "games")

    fens = process_offsets(offset, total_games, pgn)

    print("Finished making the dictionary of fens.")

    # Convert counts to probabilities
    for fen in fens:
        total = sum(fens[fen].values())
        for next_fen in fens[fen]:
            fens[fen][next_fen] /= total

    print("Finished converting counts of moves to probabilities.")

    # Save to pickle
    with open(f"data/{elo[1]}.pkl", "wb") as f:
        pickle.dump(fens, f)

    print("Finished saving to pickle.")


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
    fens = {}

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
                next_fen = board.fen()
                # Add the move to the dictionary
                if next_fen not in fens[fen]:
                    fens[fen][next_fen] = 0
                fens[fen][next_fen] += 1
            else:
                first = False
            board = game.board()
            fen = board.fen()
            if fen not in fens:
                fens[fen] = {}
            game = game.next()
        count += 1
    return fens


if __name__ == "__main__":
    path = "./lichess_2022_02.pgn"
    pgn = open(path)
    offsets, counts = get_offsets(pgn)
    for i in range(400, 3201, 400):
        process_elos((i, i + 400), offsets[i + 400], counts[i + 400])
