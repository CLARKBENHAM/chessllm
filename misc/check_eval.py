# %%
import chess
import numpy as np
from stockfish import Stockfish
from constants import STOCKFISH_PATH
from analyze import gpt_win_prob, draw_prob
from example_play import engines_play


if __name__ == "__main__":
    sf = Stockfish(STOCKFISH_PATH, {"threads": 4, "hash": 512})
    sf2 = Stockfish(STOCKFISH_PATH, {"threads": 4, "hash": 512})
    check_fens = {
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": "Board with Black's queen missing",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq -": "Board with White's queen missing",
        "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk -": "Board with a Black rook missing",
        "rnbqk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": "Board with a Black bishop missing",
        "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNK w KQkq -": "Board with a Black knight missing",
    }
    for fen, msg in check_fens.items():
        print(chess.Board(fen))
        sf.set_fen_position(fen)
        # e = sf.get_evaluation()
        # assert e["type"] == "cp"
        # cp = int(e["value"])
        # print(
        #    f"{msg}: centi-pawn eval {cp}, win prob: {gpt_win_prob(cp, 0)} draw prob:"
        #    f" {draw_prob(cp,0)}\n\n"
        # )
        sf2.set_fen_position(fen)
        results = [engines_play(sf, sf2) for _ in range(9)]
        print(msg)
        print(f"White win prob: {np.mean([i.result == 1 for i in results])}")
        print(f"White draw prob: {np.mean([i.result == 0.5 for i in results])}")
        print("\n\n")
# %%
# https://www.chess.com/article/view/the-evaluation-of-material-imbalances-by-im-larry-kaufman
# "If we make the fair but arbitrary assumption that on average the player who is behind in material has 50% compensation for it, then the rating value of a pawn (without compensation) works out to about 200 points"
# "presumably at lower levels the rating value of material is less (and at GM level more)"
# So assume 400 elo points for losing a pawn for no reason, or 1200 elo points for losing a knight for no reason
# 3700 to 3900 is a 50% chance to win, 1800-2000 is 67% to win from https://wismuth.com/elo/calculator.html#rating1=1800&rating2=2000&formula=normal

# %%
# With weaker sf I got, these win probs are crazy low and I don't believe them at all
# rn b . k b n r
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B Q K B N R
# Board with Black's queen missing: centi-pawn eval 792, win prob: 0.261 draw prob: 0.739
#
#
# r n b q k b n r
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B . K B N R
# Board with White's queen missing: centi-pawn eval -742, win prob: 0.0 draw prob: 0.782
#
#
# r n b q k b n .
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B Q K B N R
# Board with a Black rook missing: centi-pawn eval 642, win prob: 0.147 draw prob: 0.853
#
#
# r n b q k . n r
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B Q K B N R
# Board with a Black bishop missing: centi-pawn eval 585, win prob: 0.116 draw prob: 0.884
#
#
# r n b q k b . r
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B Q K B N K
# Board with a Black knight missing: centi-pawn eval -106, win prob: 0.005 draw prob: 0.982
#
#
#
