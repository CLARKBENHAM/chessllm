# %%
from collections import defaultdict
import json
from example_play import engines_play, make_engine

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
from analyze import init_df

CSV_PATH = "results/results_2023_09_22_17_47_34_587661.csv"
df = init_df(CSV_PATH)

sf, _ = make_engine()
sf2, _ = make_engine()

result_optimal = defaultdict(list)
q = df.query("result!=1 and not illegal_move.isna() and eval_type=='cp'")
x1_prob = []
x1_cp = []
y1 = []
sf.set_skill_level(20)
sf2.set_skill_level(20)
for ix, row in q.iterrows():
    print(ix, row)
    data = [engines_play(sf, sf2, [*row["moves"]]) for _ in range(30)]
    print(data)
    result_optimal[row["fen"]] += data
    print(row["gpt_win_prob"], np.sum([i.result for i in data]))
    x1_prob += [row["gpt_win_prob"]]
    x1_cp += [row["gpt_cp_result"]]
    y1 += [np.mean([i.result for i in data])]  # mean != win prob which excludes draws

# Test if win_prob is accurate
with open("best_play.txt", "w") as file:
    file.write(json.dumps(result_optimal))

# %%
with open("best_play.txt", "r") as file:
    result_optimal = json.load(file)

print(df[df["fen"].duplicated()][["fen", "ply", "gpt_cp_result", "draw_prob"]])
# %%
x1_prob = [row["gpt_win_prob"] for ix, row in q.iterrows() if row["fen"] in result_optimal]
x1_cp = [row["gpt_cp_result"] for ix, row in q.iterrows() if row["fen"] in result_optimal]

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

sns.regplot(
    x=x1_prob, y=win_p1, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2}
)
plt.title("Is win prob accurate?")
plt.xlabel("gpt win prob")
plt.ylabel("Emperical Win Percent")
corr, p = stats.pearsonr(x1_prob, win_p1)
plt.text(
    0.05,
    0.95,
    f"corr: {corr:.2f} p: {p:.2f}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
)
plt.show()
sns.regplot(x=x1_cp, y=win_p1, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2})
plt.title("Are Centi-pawns accurate?")
plt.xlabel("pred gpt win centi-pawn")
plt.ylabel("Emperical Win Percent")
corr, p = stats.pearsonr(x1_cp, win_p1)
plt.text(
    0.05,
    0.95,
    f"corr: {corr:.2f} p: {p:.2f}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
)
plt.show()
# %%
results1500 = defaultdict(list)
x2_prob = []
x2_cp = []
y2 = []
sf.set_elo_rating(1500)
sf2.set_elo_rating(1500)
for ix, row in q.iterrows():
    print(ix, row)
    data = [engines_play(sf, sf2, [*row["moves"]]) for _ in range(30)]
    print(data)
    results1500[row["fen"]] += data
    print(row["gpt_win_prob"], np.sum([i.result for i in results1500]))
    x2_prob += [row["gpt_win_prob"]]
    x2_cp += [row["gpt_cp_result"]]
    y2 += [np.mean([i.result for i in results1500])]

# Test if centi-pawn  is accurate
with open("elo_1500_play.txt", "w") as file:
    file.write(json.dumps(result_optimal))

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

sns.regplot(
    x=x2_prob, y=win_p1, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2}
)
plt.title("Is win prob accurate for 1500 elo?")
plt.xlabel("gpt win prob")
plt.ylabel("Avg Result")
corr, p = stats.pearsonr(x1_prob, y1)
plt.text(
    0.05,
    0.95,
    f"corr: {corr:.2f} p: {p:.2f}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
)
plt.show()
sns.regplot(x=x2_cp, y=win_p1, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2})
plt.title("Are Centi-pawns accurate for 1500 elo?")
plt.xlabel("gpt win centi-pawn")
plt.ylabel("Avg Result")
corr, p = stats.pearsonr(x1_prob, y1)
plt.text(
    0.05,
    0.95,
    f"corr: {corr:.2f} p: {p:.2f}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=plt.gca().transAxes,
)
plt.show()
