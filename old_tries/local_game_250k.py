"""A chess network that predicts the best move."""
import chess
import chess.pgn
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class ChessNetwork(torch.nn.Module):
    r"""A chess network that predicts the best move.

        The network takes in a 12x8x8 matrix representing the board state
        and outputs two 64-length vectors representing the probability distribution of
        each square being the best move to move from and to.
        The probability distribution of squares to move to is calculated only for the
        best square to move from.

        12x8x8 matrix:
            - 12 - one matrix 8x8 for each piece type(6 white, 6 black). It consists of
                1s and 0s, where 1s represent the presence of a piece of that type in
                that square:
                - 0-5: White pieces:
                    - 0: Pawns
                    - 1: Knights
                    - 2: Bishops
                    - 3: Rooks
                    - 4: Queens
                    - 5: King
                - 6-11: Black pieces, same as above

        White pieces are shown in the first 6 matrices, black pieces in the last 6
        If a king can castle kingside, the value is 2, if queenside, 3, if both, 4
        If a piece is the network's, it is represented by a 1, if enemy's, by -1


        The network is composed of four parts:
        - A preprocessor that takes in the board state and outputs a 128-length vector.
        This is used to extract features from the board state and is shared between the
        start and end predictors.
        - A start predictor that takes in the preprocessed board state and outputs a
        64-length vector representing the probability distribution of the best square
        to move from
        - A move preprocessor that takes in the best square to move from and outputs a
        128-length vector
        - An end predictor that takes in the preprocessed board state and the move
        preprocessor output and outputs a 64-length vector representing the probability

        The network can be shown as:
                   Input
                     |
                   Preprocessor
                 /              \\
                |                |
    Move from <-Start Predictor  |
                |                |
            Move Preprocessor    |
                \\              /
                  End Predictor -> Move to
    """

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor = torch.nn.Sequential(
            torch.nn.Conv2d(12, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 7, kernel_size=3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 8 * 8, 200),
            torch.nn.Sigmoid(),
            torch.nn.Linear(200, 128),
            torch.nn.ReLU(),
        )
        self.move_preprocessor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(128 + 64, 128),
            torch.nn.ReLU(),
        )
        self.start_predictor = torch.nn.Sequential(
            torch.nn.Linear(128, 160),
            torch.nn.Sigmoid(),
            torch.nn.Linear(160, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 64),
            torch.nn.Softmax(dim=1),
        )
        self.end_predictor = torch.nn.Sequential(
            torch.nn.Linear(256, 140),
            torch.nn.ReLU(),
            torch.nn.Linear(140, 64),
            torch.nn.Softmax(dim=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        start_sequence: torch.Tensor | None = None,
        only_start: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): The board state as a 12x8x8 matrix
            start_sequence (torch.Tensor, optional): a 1x8x8 matrix representing the
                best square to move from. Defaults to None.
            only_start (bool, optional): Whether to only return the from square.
                Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The probability distributions of the
                best square to move from and to
        """
        head = self.preprocessor(x)
        start = self.start_predictor(head)
        if start_sequence is None:
            start_sequence = torch.zeros((x.shape[0], 1, 8, 8), dtype=torch.float32).to(
                device
            )
            # Find max
            for i in range(start.shape[0]):
                start_sequence[i, 0, start[i].argmax() // 8, start[i].argmax() % 8] = 1

        if only_start:
            return start, start_sequence

        move = self.move_preprocessor(start_sequence)
        head = torch.cat((head, move), dim=1)
        return start, self.end_predictor(head)

    def train_on_a_batch(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.modules.loss._Loss,
        expected_starts: torch.Tensor,
        real_starts: torch.Tensor | None = None,
        expected_ends: torch.Tensor | None = None,
    ) -> float:
        """Train the network on a batch of data.

        Args:
            batch (torch.Tensor): The board states as a 12x8x8 matrix each
            optimizer (torch.optim.Optimizer): The optimizer to use for training
            loss_function (torch.nn.modules.loss._Loss): The loss function to use
            expected_starts (torch.Tensor): The expected start squares as a 64-length
                vectors
            real_starts (torch.Tensor, optional): The real start vector of 1x8x8
                matrices. Defaults to None.
            expected_ends (torch.Tensor, optional): The expected end squares as a
                64-length vector. Defaults to None.

        Returns:
            float: The loss
        """
        optimizer.zero_grad()
        start, end = self(batch, real_starts)
        loss = loss_function(start, expected_starts)
        if expected_ends is not None:
            loss += loss_function(end, expected_ends)
        loss.backward()
        optimizer.step()
        return loss.item()


model = ChessNetwork().to(device)
model.load_state_dict(
    torch.load(
        "250k/model_elo_all_epoch_960.pth",
        map_location=device,
    )
)  # If load existing
print(f"Num params: {sum(p.numel() for p in model.parameters())}")
print(f"Preprocessor params: {sum(p.numel() for p in model.preprocessor.parameters())}")
move_preprocessor_params = sum(p.numel() for p in model.move_preprocessor.parameters())
print(f"Move Preprocessor params: {move_preprocessor_params}")
print(f"Start params: {sum(p.numel() for p in model.start_predictor.parameters())}")
print(f"End params: {sum(p.numel() for p in model.end_predictor.parameters())}")


def board_to_matrix(board: chess.Board) -> torch.Tensor:
    """Convert a board to a 12x8x8 matrix.

    Args:
        board (chess.Board): The board to convert

    Returns:
        torch.Tensor: The board as a 12x8x8 matrix
    """
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
    matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
    return matrix


def ask_ai_for_moves(matrix: torch.Tensor) -> list[tuple[float, str]]:
    """Ask the AI for a move.

    Args:
        matrix (torch.Tensor): Board matrix

    Returns:
        list[tuple[float, str]]: List of probabilities and corresponding moves
    """
    start, end = model(matrix)
    start = start.squeeze(0).detach().cpu().numpy()
    end = end.squeeze(0).detach().cpu().numpy()
    # Get all possible moves with their probabilities
    legal = [legal.uci() for legal in board.legal_moves]
    # If verbose, what illegal wanted
    moves = [
        (start[i] * end[j], chess.SQUARE_NAMES[i] + chess.SQUARE_NAMES[j])
        for i in range(64)
        for j in range(64)
    ]
    # Sort moves by probability
    moves.sort(key=lambda x: x[0], reverse=True)
    move = moves[0]
    while move[1] not in legal:
        print(f"Move {move[1]} not legal")
        moves.pop(0)
        move = moves[0]
    # moves = [
    #     (start[i] * end[j], chess.SQUARE_NAMES[i] + chess.SQUARE_NAMES[j])
    #     for i in range(64)
    #     for j in range(64)
    #     if chess.SQUARE_NAMES[i] + chess.SQUARE_NAMES[j] in legal
    # ]
    # # Sort moves by probability
    # moves.sort(key=lambda x: x[0], reverse=True)
    return moves


def ask_ai_for_move(matrix: torch.Tensor) -> str:
    """Ask the AI for a move.

    Args:
        matrix (torch.Tensor): Board

    Returns:
        str: Move in UCI format
    """
    return ask_ai_for_moves(matrix)[0][1]


def ask_ai_thoughts(matrix: torch.Tensor, move: str) -> float:
    """Get AI's thoughts on a move.

    Get all start probabilities. Get end probabilities for a move.
    Return their product for a move.

    Args:
        matrix (torch.Tensor): Board matrix
        move (str): Move in UCI format

    Returns:
        float: Probability of move
    """
    start, _ = model(matrix, only_start=True)
    start = start.squeeze(0).detach().cpu().numpy()
    start = start[chess.SQUARE_NAMES.index(move[:2])]
    move_matrix = torch.zeros((matrix.shape[0], 1, 8, 8), dtype=torch.float32).to(
        device
    )
    move_matrix[
        0,
        0,
        chess.SQUARE_NAMES.index(move[2:4]) // 8,
        chess.SQUARE_NAMES.index(move[2:4]) % 8,
    ] = 1
    _, end = model(matrix, move_matrix)
    end = end.squeeze(0).detach().cpu().numpy()
    end = end[chess.SQUARE_NAMES.index(move[2:4])]
    return start * end


def ask_human_move(board: chess.Board) -> str:
    """Ask human for a move.

    Args:
        board (chess.Board): Board

    Returns:
        str: Move in UCI format
    """
    # Get legal moves in Algebraic Notation
    legal_moves = [board.san(x) for x in board.legal_moves]
    while True:
        print(board)
        move_str = input("Enter move: ")
        # Check whether the move is legal
        if move_str in legal_moves or move_str == "resign":
            break
        print("Illegal move")
    return move_str


if __name__ == "__main__":
# Create board from fen
    fen = input("Enter FEN: ")
    board = chess.Board(fen)

    while True:
        # Ask for human move
        # move_str = ask_human_move(board)
        # try:
        #     print("Your move:", move_str)
        #     board.push_san(move_str)
        # except ValueError:
        #     print("Resigning")
        #     pgn = chess.pgn.Game.from_board(board)
        #     print()
        #     print()
        #     print(pgn)
        #     break

        try:
            matrix = board_to_matrix(board)
            move_str = ask_ai_for_move(matrix)
            print("AI's move:", move_str)
            # Ask AI's thoughts on move
            # print(
            #     "AI's thoughts on",
            #     move_str,
            #     "are",
            #     ask_ai_thoughts(matrix, move_str),
            # )
            board.push_uci(move_str)
        except IndexError:
            print("End of game")
            pgn = chess.pgn.Game.from_board(board)
            print()
            print()
            print(pgn)
            break
        except Exception as e:
            pgn = chess.pgn.Game.from_board(board)
            print()
            print()
            print(pgn)
            print(e)
            break
        except KeyboardInterrupt:
            pgn = chess.pgn.Game.from_board(board)
            print()
            print()
            print(pgn)
            break
