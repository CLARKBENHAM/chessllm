WIN_CUTOFF = 300  # in centi-pawns, should go down as models get stronger
# Stockfish.get_evaluation() return the normalized export: https://github.com/official-stockfish/Stockfish/blob/70ba9de85cddc5460b1ec53e0a99bee271e26ece/src/uci.cpp#L317
# But other functions act on a raw value with is used to compute winning chances

NormalizeToPawnValue = 328
# Values tied to win percentage implemented below
# https://github.com/official-stockfish/Stockfish/blob/70ba9de85cddc5460b1ec53e0a99bee271e26ece/src/uci.cpp#L209
# https://stockfishchess.org/blog/2022/stockfish-15-1/#:~:text=With%20a%20%2B1%20evaluation%2C%20Stockfish%20has%20now%20a%2050%25%20chance%20of%20winning%20the%20game%20against%20an%20equally%20strong%20opponent.
# They assume a 50% win rate at +1 value
# To get CP score the return  (value * 100 / NormalizeToPawnValue)
# https://github.com/official-stockfish/Stockfish/blob/22cdb6c1ea1f5ca429333bcbe26706c8b4dd38d7/src/search.cpp#L1905C1-L1906C1

VALUE_MATE_IN_MAX_PLY = 31754  # taken straight from stockfish
NORMALIZED_SCORE_MAX = VALUE_MATE_IN_MAX_PLY * 100 / NormalizeToPawnValue
WIN_CP = NORMALIZED_SCORE_MAX

# from VALUE_MATE https://github.com/official-stockfish/Stockfish/blob/22cdb6c1ea1f5ca429333bcbe26706c8b4dd38d7/src/types.h#L164
# convert to centi-pawns
STOCKFISH_PATH = "/usr/local/bin/stockfish"
