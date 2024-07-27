from typing import Iterator, TextIO

import chess
import chess.pgn
import yaml


def load_offsets(
    pgn: TextIO,
    analyzed_games_per_rating: int = 6000,
    max_analyzed_games: int = 500000,
    categories=[
        400,
        800,
        1200,
        1600,
        2000,
        2400,
        2800,
        3200,
    ],
) -> dict[tuple[int, int], list[int]]:
    rating_based_offsets: dict[tuple[int, int], list[int]] = {}
    for i in range(len(categories)):
        rating_based_offsets[(([0] + categories)[i], ([0] + categories)[i + 1])] = []

    count = 0
    while count < max_analyzed_games:
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)
        if headers is None or all(
            len(rating) >= analyzed_games_per_rating
            for rating in rating_based_offsets.values()
        ):
            break

        elos = headers.get("WhiteElo"), headers.get("BlackElo")
        if any(not (i and i.isnumeric()) for i in elos):
            continue

        elos = tuple(int(i or 0) for i in elos)
        for rating_range in rating_based_offsets:
            if any(rating_range[0] <= elo <= rating_range[1] for elo in elos):
                if len(rating_based_offsets[rating_range]) >= analyzed_games_per_rating:
                    break
                rating_based_offsets[rating_range].append(offset)
                break

        count += 1
        if not count % 1000:
            loaded = sum(len(x) for x in rating_based_offsets.values())
            print(
                f"Loaded {loaded}/{analyzed_games_per_rating * len(categories)} games",
                f"({count}/{max_analyzed_games} analyzed)",
                end="\r",
            )

    return rating_based_offsets


def load_offset(pgn: TextIO, offset: int, moves: dict[str, list[str]]) -> None:
    pgn.seek(offset)
    game_start = chess.pgn.read_game(pgn)
    if not game_start or game_start.errors:
        return
    board = game_start.board()
    prev_fen = board.fen(en_passant="fen")
    game = game_start.next()
    while game:
        board = game.board()
        fen = board.fen(en_passant="fen")
        move = str(game.san())
        if prev_fen in moves:
            moves[prev_fen].append(move)
        else:
            moves[prev_fen] = [move]
        prev_fen = fen
        game = game.next()


def load_dataset(
    path: str, analyzed_games_per_rating: int = 6000
) -> Iterator[tuple[tuple[int, int], dict[str, list[str]]]]:
    pgn = open(path)

    rating_based_offsets = load_offsets(pgn, analyzed_games_per_rating)
    print()
    print("Loaded offsets")
    for block_id, (rating_range, offsets) in enumerate(rating_based_offsets.items()):
        rating_moves: dict[str, list[str]] = {}
        for game_id, offset in enumerate(offsets):
            load_offset(pgn, offset, rating_moves)
            if not game_id % 100:
                print(
                    f"{game_id}/{len(offsets)} games from block {block_id}/{len(rating_based_offsets)}",
                    end="\r",
                )
        print(
            f"Loaded {len(offsets)} games from block {block_id}/{len(rating_based_offsets)}"
        )
        yield rating_range, rating_moves


def dump_dataset(
    path_prefix: str, dataset: Iterator[tuple[tuple[int, int], dict[str, list[str]]]]
):
    for i, (rating_range, moves) in enumerate(dataset):
        print(f"Dumping block {i}")
        with open(
            f"{path_prefix}_{rating_range[0]}_{rating_range[1]}.yaml",
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(moves, f)


def main():
    path = "../lichess_2022_02.pgn"
    prefix = "lichess_2022_02_partition"
    dataset = load_dataset(path)
    dump_dataset(prefix, dataset)


if __name__ == "__main__":
    main()
