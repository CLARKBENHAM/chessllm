WIN_CUTOFF = 300  # in centi-pawns, tied to win percentage implemented below
# https://github.com/official-stockfish/Stockfish/blob/70ba9de85cddc5460b1ec53e0a99bee271e26ece/src/uci.cpp#L209
# https://stockfishchess.org/blog/2022/stockfish-15-1/#:~:text=With%20a%20%2B1%20evaluation%2C%20Stockfish%20has%20now%20a%2050%25%20chance%20of%20winning%20the%20game%20against%20an%20equally%20strong%20opponent.
WIN_CP = 200 * 100
STOCKFISH_PATH = "/usr/local/bin/stockfish"
