import torch


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


class ChessNetwork(torch.nn.Module):
    r"""A chess network that predicts the best move.

    The network takes in a 24x8x8 matrix representing the board state
    before and after a move and outputs a number in [0,1] - how good the move is

    24x8x8 matrix:
        - 12 - one matrix 8x8 for each piece type: 6 white, 6 black
            (Then doubled to 24, because 12 for the board before the move and 12 for the board after the move).
            It consists of 1s and 0s, where 1s represent the presence of a piece of that type in
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
    """

    def __init__(self) -> None:
        super().__init__()
        self.processor = torch.nn.Sequential(
            torch.nn.Conv2d(24, 200, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(200, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 7, kernel_size=3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 8 * 8, 200),
            torch.nn.Sigmoid(),
            torch.nn.Linear(200, 70),
            torch.nn.ReLU(),
            torch.nn.Linear(70, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): 24x8x8 tensor representing the board state
                before and after a move

        Returns:
            torch.Tensor: 1x1 tensor representing how good the move is
        """
        return self.processor(x)

    def train_on_a_batch(
        self,
        input_batch: torch.Tensor,
        output_batch: torch.Tensor,
        optimiser: torch.optim.Optimizer,
        loss_function: torch.nn.modules.loss._Loss,
    ) -> float:
        """Trains the network on a batch of data.

        Uses expected starts and ends distribution to calculate
        how good the move is according to dataset
        by multiplying the probabilities of the start and end squares
        for each board position.

        Args:
            batch (list[tuple[str, str]]): Batch of fens before and after the move
            optimiser (torch.optim.Optimizer): Optimiser used for training
            loss_function (torch.nn.modules.loss._Loss): Loss function used for training

        Returns:
            float: Loss on the batch
        """
        optimiser.zero_grad()
        outputs = self(input_batch)
        loss = loss_function(outputs, output_batch)
        loss.backward()
        optimiser.step()
        return loss.item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
NAME = "269eval"
model = ChessNetwork().to(device)
model.load_state_dict(
    torch.load(
        f"{NAME}/model_800_epoch_0_batch_10000.pth",
        map_location=device,
    )
)  # If load existing
print(f"Num params: {sum(p.numel() for p in model.parameters())}")
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()


from_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
to_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

output = model(fens_to_tensor((from_fen, to_fen)))
