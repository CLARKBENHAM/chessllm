# %%
import sys

from collections import defaultdict
import json


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import warnings
from itertools import zip_longest
import math

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns

from stockfish import Stockfish
import ast
from constants import STOCKFISH_PATH, WIN_CUTOFF, WIN_CP, NormalizeToPawnValue
from analyze import init_df, reg_plot
from example_play import engines_play, make_engines, GameResults

CSV_PATH = "results/results_2023_09_22_17_47_34_587661.csv"
df = init_df(CSV_PATH)

sf, _ = make_engines()
sf2, _ = make_engines()
# %%
result_optimal = defaultdict(list)
q = df.query("result!=1 and not illegal_move.isna() and eval_type=='cp'")
y1 = []
sf.set_skill_level(20)
sf2.set_skill_level(20)
for ix, row in q.iterrows():
    print(ix, row)
    data = [engines_play(sf, sf2, [*row["moves"]]) for _ in range(30)]
    print(data)
    result_optimal[row["fen"]] += data
    print(row["gpt_win_prob"], np.sum([i.result for i in data]))
    y1 += [np.mean([i.result for i in data])]  # mean != win prob which excludes draws

# Test if win_prob is accurate
with open("results/best_play.txt", "w") as file:
    file.write(json.dumps(result_optimal))

# %%
with open("results/best_play.txt", "r") as file:
    result_optimal = json.load(file)
    for k, v in result_optimal.items():
        result_optimal[k] = [GameResults(*i) for i in v]

print(df[df["fen"].duplicated()][["fen", "ply", "gpt_cp_result", "draw_prob"]])
# %%
x1_pred_win = [row["gpt_win_prob"] for ix, row in q.iterrows() if row["fen"] in result_optimal]
x1_pred_draw = [row["draw_prob"] for ix, row in q.iterrows() if row["fen"] in result_optimal]
x1_pred_cp = [row["gpt_cp_result"] for ix, row in q.iterrows() if row["fen"] in result_optimal]

win_p1 = [
    np.mean(
        [
            i.result == 1 if row["white"] != "stockfish" else i.result == 0
            for i in result_optimal[row["fen"]]
        ]
    )
    for ix, row in q.iterrows()
    if row["fen"] in result_optimal
]

draw_p1 = [
    np.mean([i.result == 0.5 for i in result_optimal[row["fen"]]])
    for ix, row in q.iterrows()
    if row["fen"] in result_optimal
]


reg_plot(
    x1_pred_win, win_p1, "gpt win prob", "Emperical Win Percent", title="Is win prob accurate?"
)
reg_plot(
    x1_pred_cp,
    win_p1,
    "gpt win centi-pawn",
    "Emperical Win Percent",
    title="Are Centi-pawns accurate?",
)
reg_plot(
    x1_pred_draw, draw_p1, "draw prob", "Emperical Draw Percent", title="Is Draw prob accurate?"
)
reg_plot(
    x1_pred_cp,
    draw_p1,
    "gpt win centi-pawn",
    "Emperical Draw Percent",
    title="Are Centi-pawns accurate for Draws?",
)

ep = 1e-6
_to_log = lambda v: np.log(max(ep, min(1 - ep, v)))
_take_log = lambda l: list(map(_to_log, l))
_take_log_odds = lambda l: list(map(lambda v: _to_log(v / (1 + ep - v)), l))
print_logs = False
if print_logs:
    reg_plot(
        _take_log(x1_pred_win),
        _take_log(win_p1),
        "Log gpt pred win prob",
        "Log real win rate",
    )
    reg_plot(_take_log(x1_pred_win), win_p1, "Log gpt pred win prob", "real win rate")
    reg_plot(x1_pred_cp, _take_log(win_p1), "gpt cp eval", "Log real win rate")
    reg_plot(x1_pred_cp, _take_log_odds(win_p1), "gpt cp eval", "Log Odds Real win rate")


# %%
result_1500 = defaultdict(list)
y2 = []
sf.set_elo_rating(1500)
sf2.set_elo_rating(1500)
for ix, row in q.iterrows():
    print(ix, row)
    data = [engines_play(sf, sf2, [*row["moves"]]) for _ in range(30)]
    print(data)
    result_1500[row["fen"]] += data
    print(
        row["gpt_win_prob"],
        np.mean([i.result == (1 if "gpt" in row["white"] else 0) for i in data]),
    )

    y2 += [np.mean([i.result == (1 if "gpt" in row["white"] else 0) for i in data])]

# Test if centi-pawn  is accurate
with open("elo_1500_play.txt", "w") as file:
    file.write(json.dumps(result_1500))

# %%
with open("results/elo_1500_play.txt", "r") as file:
    result_1500 = json.load(file)
    for k, v in result_1500.items():
        result_1500[k] = [GameResults(*i) for i in v]

# %%
x2_pred_win = [row["gpt_win_prob"] for ix, row in q.iterrows() if row["fen"] in result_1500]
x2_pred_draw = [row["draw_prob"] for ix, row in q.iterrows() if row["fen"] in result_1500]
x2_pred_cp = [row["gpt_cp_result"] for ix, row in q.iterrows() if row["fen"] in result_1500]
win_p2 = [
    np.mean(
        [
            i.result == 1 if row["white"] != "stockfish" else i.result == 0
            for i in result_1500[row["fen"]]
        ]
    )
    for ix, row in q.iterrows()
    if row["fen"] in result_1500
]

draw_p2 = [
    np.mean([i.result == 0.5 for i in result_1500[row["fen"]]])
    for ix, row in q.iterrows()
    if row["fen"] in result_1500
]

reg_plot(
    x2_pred_win,
    win_p2,
    "gpt win prob",
    "Emperical Win Percent",
    title="Is Win Prob accurate for 1500 elo?",
)
reg_plot(
    x2_pred_cp,
    win_p2,
    "gpt win centi-pawn",
    "Emperical Win Percent",
    title="Are Centi-pawns accurate for Wins for 1500 elo?",
)
reg_plot(
    x2_pred_draw,
    draw_p2,
    "draw prob",
    "Emperical Draw Percent",
    title="Is Draw prob accurate for 1500 elo?",
)
reg_plot(
    x2_pred_cp,
    draw_p2,
    "gpt win centi-pawn",
    "Emperical Draw Percent",
    title="Are Centi-pawns accurate for Draws for 1500 elo?",
)

# Using the default scaling: For optimal play Win Prob is 0.94 , 0.89 for Draw Prob. For elo 1500 Win Prob is 0.81 and Draw Prob is 0.29
# Using CP as straight values: For optimal play Win Prob is 0.84 , 0.79 for Draw Prob. For elo 1500 Win Prob is 0.82 and Draw Prob is 0.58
# I think I'll keep the default scaling
#

# %%
sf.set_elo_rating(1500)


def _get_weak_eval(white, fen):
    sf.set_fen_position(fen)
    v = sf.get_evaluation()["value"]
    return v if white != "stockfish" else -v


elo1500_sf_pred_cp = [
    _get_weak_eval(white, fen) for ix, (white, fen) in q[["white", "fen"]].iterrows()
elo1500_sf_pred_win = [gpt_win_prob(cp, ply) for cp, ply in zip(elo1500_sf_pred_cp, q["ply"])]

reg_plot(
    elo1500_sf_pred_win,
    win_p2,
    "gpt win prob by elo 1500 estimator",
    "Emperical Win Percent",
    title="Is Win Prob accurate for 1500 elo?",
)
reg_plot(
    elo1500_sf_pred_cp,
    win_p2,
    "gpt win prob by elo 1500 estimator",
    "Emperical Win Percent",
    title="Is Win Prob accurate for 1500 elo?",
)
# %%


def cross_entropy(p, q):
    pq = p * np.log(q)
    ce = -np.sum(pq)
    return ce


def gpt_win_prob_validate(
    gpt_cp_result=None, ply=None, normalize_const=NormalizeToPawnValue / 100, **kwargs
):
    """
    Gets the probability the most powerful stockfish version would win, given an evaluation.
        This will be incorrect for low level play:  says '3rr3/1p3pk1/pqb1pn1p/6p1/7P/1NPB1QR1/P1P2PP1/4R1K1 w - - 3 24'
        a 201cp lead is a 99.7% win chance but in the experiment it wins 4/9 at elo=1200 and 3/9 at elo=1500
        ```
        >>>uci_moves = df.query("gpt_win_prob>0.99 and gpt_cp_result < 300").iloc[0]["moves"]
        >>>sf, _ = make_engines()
        >>>sf2, _ = make_engines()
        >>>sf.set_position(uci_moves)
        >>>sf2.set_position(uci_moves)
        >>>results = [engines_play(sf, sf2, [*uci_moves]) for _ in range(9)]
        >>>[i.result for i in results]
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        # 4 wins, 5 losses
        ```
        With a skill level of 20 it was np.unique([i.result for i in results3], return_counts=True)==(array([0.5, 1. ]), array([122, 287]))
        Which should be a prob of (287+0.5*122)/(122+287) = 85% with elo advancement, or 70% outright win
    Stockfish returns it's value * 100 / UCI::NormalizeToPawnValue (328) to convert to centi-pawns
    https://github.com/official-stockfish/Stockfish/blob/22cdb6c1ea1f5ca429333bcbe26706c8b4dd38d7/src/uci.cpp#L315
    But the win_rate_model is based on the original value: https://github.com/official-stockfish/Stockfish/blob/70ba9de85cddc5460b1ec53e0a99bee271e26ece/src/uci.cpp#L209
    And https://github.com/official-stockfish/WDL_model/blob/6977d86df30e3529875c471336abe723afb018c9/scoreWDL.py#L53C11-L53C11
        says cp to value is v * NormalizeToPawnValue / 100

    gpt_cp_result: the evaluation of the board in *centipawns*, in favor of GPT.
        The docs say a "value" of +1 is 50% chance of winning, but convert to centipawns when printing the evaluation
        https://github.com/official-stockfish/Stockfish/blob/22cdb6c1ea1f5ca429333bcbe26706c8b4dd38d7/src/search.cpp#L1905C1-L1906C1
    ply is number of total moves
    returns win probability : 1- lose probability - draw probability.
        win and lose probability can both be 0
    """
    value = gpt_cp_result * normalize_const
    # The model only captures up to 240 plies, so limit the input and then rescale
    m = min(240, ply) / 64.0

    # The coefficients of a third-order polynomial fit is based on the fishtest data
    # for two parameters that need to transform eval to the argument of a logistic
    # function.
    as_ = [0.38036525, -2.82015070, 23.17882135, 307.36768407]
    bs_ = [-2.29434733, 13.27689788, -14.26828904, 63.45318330]

    # Enforce that NormalizeToPawnValue corresponds to a 50% win rate at ply 64
    assert NormalizeToPawnValue == int(
        as_[0] + as_[1] + as_[2] + as_[3]
    ), f"{NormalizeToPawnValue} == {int(as_[0] + as_[1] + as_[2] + as_[3])}"

    a = (((as_[0] * m + as_[1]) * m + as_[2]) * m) + as_[3]
    b = (((bs_[0] * m + bs_[1]) * m + bs_[2]) * m) + bs_[3]

    # Transform the eval to centipawns with limited range
    x = max(-4000.0, min(4000.0, float(value)))

    # Return the win rate in per mille units rounded to the nearest value
    mp = int(0.5 + 1000 / (1 + math.exp((a - x) / b)))
    return mp / 1000.0


def calc_entropy(results_dict, name, normalize_const=None, display=False):
    pred_p = []
    obv_p = []
    for fen, plays in results_dict.items():
        pred_p += [
            df[df["fen"] == fen]
            .apply(lambda r: gpt_win_prob_validate(**r, normalize_const=normalize_const), axis=1)
            .mean()
        ]
        # true_p += [i.result == 1 if row["white"] != "stockfish" else i.result == 0 for i in plays]
        obv_p += [
            np.mean(
                [i.result == 1 if row["white"] != "stockfish" else i.result == 0 for i in plays]
            )
        ]
    n_samples = np.ceil(np.mean(list(map(len, results_dict.values()))))
    # the epsilon that has a 95% chance of observing 0 with n_samples
    e = min(np.logspace(-12, -1), key=lambda p: abs(0.95 - stats.binom.cdf(0.0, n_samples, p)))
    pred_p = np.clip(pred_p, e, 1 - e)
    obv_p = np.clip(obv_p, e, 1 - e)

    # Compute Cross-entropy of est_p and true_p
    out = {
        "pred_entropy": stats.entropy(pred_p),
        "obv_entropy": stats.entropy(obv_p),
        "kl_divergence": stats.entropy(obv_p, pred_p),
        "reverse_kl_divergence": stats.entropy(pred_p, obv_p),
        "cross_entropy": cross_entropy(pred_p, obv_p),
    }
    if display:
        if normalize_const is not None:
            name = f"{name} normalized by {normalize_const}"
        print(name)
        print(f"Pred Entropy: {out['pred_entropy']}")
        print(f"Observed Entropy: {out['obv_entropy']}")
        print(f"Kullback-Leibler divergence: {out['kl_divergence']}")
        print(f"Reverse Kullback-Leibler divergence: {out['reverse_kl_divergence']}")
        print(f"Cross Entropy: {out['cross_entropy']}")
        print()
    return out


ncs = np.linspace(0.1, 10, 50)
optimal_e = pd.DataFrame(
    [calc_entropy(result_optimal, "optimal", normalize_const=nc) for nc in ncs]
)
elo1500_e = pd.DataFrame([calc_entropy(result_1500, "elo_1500", normalize_const=nc) for nc in ncs])
# %%
print(
    "Normalizing constant minimizing KL for optimal play: ",
    ncs[optimal_e["kl_divergence"].argmin()],
)
print(
    "Normalizing constant minimizing KL for elo1500 play: ",
    ncs[elo1500_e["kl_divergence"].argmin()],
)

# %%
# Entropy is minimize when win_p is 0 or 1, there's not much difference between the 2 distributions
plt.title("Entropy vs Normalization Constant for Optimal Play")
for col in optimal_e.columns:
    plt.plot(ncs, optimal_e[col], label=col)
plt.legend()
plt.show()

plt.title("Entropy vs Normalization Constant for Elo1500 Play")
for col in elo1500_e.columns:
    plt.plot(ncs, elo1500_e[col], label=col)
plt.legend()
plt.show()

# optimal
# Est Entropy: 2.3568193396283057
# True Entropy: 3.357234463268411
# Kullback-Leibler divergence: 2.8398702301534566
# Reverse Kullback-Leibler divergence: 5.513588077122282
# Cross Entropy: 20.125480500476435
#
# elo_1500
# Est Entropy: 2.7187269892934727
# True Entropy: 4.184310096051774
# Kullback-Leibler divergence: 2.402967683214135
# Reverse Kullback-Leibler divergence: 5.064896116000756
# Cross Entropy: 21.333153601901977
#
