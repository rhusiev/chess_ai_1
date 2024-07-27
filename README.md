# Chess neural network

## New 266k model

## Old 504k model

### First 500 epochs

lichess_2022_02.pgn
`games_to_matrices.py`:
\- `GAMES_START` = 0
\- `GANES_END` = 500000

### Second 500 epochs

`lichess_2022_02.pgn`
`games_to_matrices.py`:
\- `GAMES_START` = 500000
\- `GANES_END` = 1000000

### Next 250 epochs

`lichess_2022_02.pgn`:
\- Take both 0 to 500000 and 500000 to 1000000

### Next 250 epochs

`lichess_2022_02.pgn` and `chess_tactics.pgn`:
\- Take both 0 to 500000 and 500000 to 1000000
\- Also take everything from chess_tactics.pgn

### Next 500 epochs

Just `chess_tactics.pgn`:
\- Take everything from chess_tactics.pgn