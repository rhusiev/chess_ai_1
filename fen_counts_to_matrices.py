import pickle as pkl
from itertools import islice

import chess
import chess.pgn
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def fen_to_matrix(fen: str):
    board = chess.Board(fen)
    color = board.turn
    matrix = torch.zeros((12, 8, 8), dtype=torch.float32, device=device).to(device)
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
    return matrix


def fens_to_tensor(fens: tuple[str, str]):
    return torch.cat((fen_to_matrix(fens[0]), fen_to_matrix(fens[1]))).to(device)


def batch_to_tensor(batch):
    return torch.stack([fens_to_tensor(fens) for fens in batch]).to(device)


def batch_to_tensors(
    batch: list[tuple[str, dict[str, float]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    board_batch = [chess.Board(fen[0]) for fen in batch]
    moves_batch = []
    probabilities_batch = []
    for i, board in enumerate(board_batch):
        start_fen = board.fen()
        legal_moves = board.legal_moves
        new_moves = []
        new_probabilities = []
        existing_fens_probabilities = batch[i][1].items()
        existing_fens = [fen[0] for fen in existing_fens_probabilities]
        for fen in existing_fens_probabilities:
            new_moves.append((start_fen, fen[0]))
            new_probabilities.append(fen[1])
        count = 0
        for move in legal_moves:
            board.push(move)
            fen = board.fen()
            if fen in existing_fens:
                continue
            new_moves.append((start_fen, fen))
            new_probabilities.append(0.0)
            if count >= 1:
                break
            count += 1
            board.pop()
        moves_batch.extend(new_moves)
        probabilities_batch.extend(new_probabilities)
    matrices = batch_to_tensor(moves_batch)
    probabilities_batch = (
        torch.tensor(probabilities_batch)
        .reshape(len(probabilities_batch), 1)
        .to(device)
    )
    return matrices, probabilities_batch


for elo in range(800, 2801, 400):
    with open(f"data/{elo}.pkl", "rb") as f:
        fens = pkl.load(f).items()
    print(f"Loaded {len(fens)} fens for elo {elo}")
    input_tensor, output_tensor = batch_to_tensors(fens)
    torch.save(input_tensor, f"data/{elo}_input.pt")
    torch.save(output_tensor, f"data/{elo}_output.pt")
    print(f"Saved tensors for elo {elo}")
