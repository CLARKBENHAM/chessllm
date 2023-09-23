# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import warnings
from itertools import zip_longest

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns

from stockfish import Stockfish
import ast
from constants import STOCKFISH_PATH, WIN_CUTOFF, WIN_CP


def try_ast_eval(s):
    try:
        return ast.literal_eval(s)
    except:
        return pd.NA


# trend towards AI playing better against better models
CSV_PATH = "results/results_2023_09_22_13_39_17_831054.csv"
CSV_PATH = "results/results_2023_09_22_15_47_34_587661.csv"  # no trend
CSV_PATH = "results/results_2023_09_22_17_01_40_144335.csv"

df = pd.read_csv(
    CSV_PATH,
    converters={"moves": ast.literal_eval, "eval": try_ast_eval},
    # errors="coherce",
)

# evals is NA if no illegal move made
assert (df["illegal_move"].isna() == df["eval"].isna()).all()

df["eval_type"] = df["eval"].apply(lambda e: e["type"] if isinstance(e, dict) else pd.NA)
df["eval_value"] = (
    df["eval"].apply(lambda e: e["value"] if isinstance(e, dict) else pd.NA).astype("Int64")
)


# Set positive numbers are better for gpt
df["eval_value"].loc[df["white"] == "stockfish"] *= -1
# df["result"].loc[df["white"] == "stockfish"] = 1 - df["result"].loc[df["white"] == "stockfish"]
df.loc[df["white"] == "stockfish", "result"] = 1 - df["result"].loc[df["white"] == "stockfish"]


# %%
def decide_game(white=None, result=None, moves=None, eval_type=None, eval_value=None, **kwargs):
    """A numerical representation of the games result, according to stockfish's eval.
    positive is in favor of gpt, negative is in favor of stockfish
    """
    if pd.isna(eval_type):
        val = result
        if val > 0.5:
            val = WIN_CP
        elif val < 0.5:
            val = -WIN_CP
    elif eval_type == "mate":
        if eval_value == 0:  # mate is 0 for both win and lose
            white_won = len(moves) % 2 == 1
            val = WIN_CP if white_won else -WIN_CP
        else:
            val = WIN_CP if eval_value > 0 else -WIN_CP
    else:
        val = eval_value
    is_gpt = white != "Stockfish"
    val = val if is_gpt else -val
    return val


df["gpt_cp_result"] = df.apply(lambda row: decide_game(**row), axis=1)

illegal_p = df["illegal_move"].notna().mean()
win_p = (
    (df["result"] == 1)  # won
    | (df.loc[df["eval_type"] == "mate", "eval_value"] > 0)  # had a forced mate
    | (df.loc[df["eval_type"] == "cp", "eval_value"] >= WIN_CUTOFF)  # had a big lead
).mean()

draw_p = (
    (df["result"] == 0.5)
    | (
        (-WIN_CUTOFF < df.loc[df["eval_type"] == "cp", "eval_value"])
        & (df.loc[df["eval_type"] == "cp", "eval_value"] < WIN_CUTOFF)
    )
).mean()
loss_p = (
    (df["result"] == 0)
    & (
        df["eval_type"].isna()
        | (df.loc[df["eval_type"] == "mate", "eval_value"] < 0)
        | (df.loc[df["eval_type"] == "cp", "eval_value"] <= -WIN_CUTOFF)
    )
).mean()
print(
    f"Percent GPT ended game by illegal move: {illegal_p:.0%} ",
)
print(
    f"GPT win/loss/draw percentages: {win_p:.0%}, {loss_p:.0%}, {draw_p:.0%} ",
)
assert 1 == win_p + loss_p + draw_p, win_p + loss_p + draw_p
assert (
    win_p == (df["gpt_cp_result"] >= WIN_CUTOFF).mean()
    and loss_p == (df["gpt_cp_result"] <= -WIN_CUTOFF).mean()
    and draw_p == (df["gpt_cp_result"].abs() < WIN_CUTOFF).mean()
)
plt.title("Number of Forced mates after invalid moves for GPT")
plt.hist(
    df[df["eval_type"] == "mate"].apply(
        lambda i: -i["eval_value"] if i["white"] == "stockfish" else i["eval_value"], axis=1
    ),
    bins=30,
)
plt.show()

# %% plot distribution of gpt evals where games ended early, by elo bin
n_elo_bins = 3
min_elo_width = 50
mn, mx = df["white_elo"].min(), df["white_elo"].max()
n_elo_bins = min(n_elo_bins, int((mx - mn) // min_elo_width))
elo_bins = np.arange(mn, mx + 1, (mx - mn) / (n_elo_bins + 1))
fig, axs = plt.subplots(n_elo_bins, 1, sharex=True, sharey=True)
fig.set_size_inches(12, 8)
fig.suptitle("Evaluation of invalid moves")
easiest_data = df.loc[
    (df["eval_type"] == "cp") & (df["white_elo"] >= elo_bins[0]) & (df["white_elo"] < elo_bins[1]),
    "gpt_cp_result",
]

for ax, (min_elo, max_elo) in zip(axs, zip_longest(elo_bins[:-1], elo_bins[1:], fillvalue=mx + 1)):
    data = df.loc[
        (df["eval_type"] == "cp") & (df["white_elo"] >= min_elo) & (df["white_elo"] < max_elo),
        "gpt_cp_result",
    ]
    ax.hist(data, bins=30, weights=np.ones(len(data)) / len(data))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    title = f"Elo: {min_elo:.0f}-{max_elo:.0f}"
    ks_stat, p_value = stats.ks_2samp(data.values, easiest_data.values)
    if p_value < 0.1:
        title += f" KS: {ks_stat:.2f} p: {p_value:.2f} "
    ax.set_title(title)
    ax.vlines(0, 0, 1, color="black", linewidth=0.5)
fig.subplots_adjust(hspace=0.5)
fig.show()
# %%  plot length of game versus elo
x = df["white_elo"].values
y = df["moves"].apply(len).values
sns.regplot(x=x, y=y, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2})
plt.title("Length of game versus Stockfish Elo")
plt.xlabel("Elo")
plt.ylabel("Number of moves")
corr, p = stats.pearsonr(x, y)
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
# graph probablity that gpt wins vs elo of gpt.
# How to convert evaluation to probability of winning?


def stockfish_win_prob(eval):
    """
    implementation of https://github.com/official-stockfish/Stockfish/blob/70ba9de85cddc5460b1ec53e0a99bee271e26ece/src/uci.cpp#L209
    """


# %% For each entry in the df parse the moves column, a list of moves and return the stockfish evaluation of the board for that move. The result should be a series of lists
# Very Slow
sf = Stockfish(STOCKFISH_PATH, parameters={"Threads": 6, "Hash": 512, "Skill Level": 20})


def get_evals(white, moves, check_every=4):
    "slow!"
    evals = []
    is_gpt = white != "Stockfish"
    end = [len(moves)] if len(moves) % check_every != 0 else []
    for i in list(range(0, len(moves), check_every)) + end:
        sf.set_position(moves[:i])
        e = sf.get_evaluation()  # in terms of white's position
        # a draw is {'type': 'cp', 'value': 0}
        val = e["value"] if is_gpt else -e["value"]
        if e["type"] == "mate":
            if val == 0:  # mate is 0 for both win and lose
                white_won = i % 2 == 1
                val = WIN_CP if white_won == is_gpt else -WIN_CP
            else:
                val = WIN_CP if val > 0 else -WIN_CP
        evals += [val]
    return np.array(evals)


n_games_graph = 10
df["ts_evals"] = df.iloc[:n_games_graph].apply(lambda r: get_evals(r["white"], r["moves"]), axis=1)

plt.title("Evaluation time series for GPT")
plt.gca().set_ylim([-2 * WIN_CUTOFF, 2 * WIN_CUTOFF])
plt.gca().set_xlabel("Move Number")
plt.gca().set_ylabel("Centipawns")
# Against the index of moves, plot the evaluation of the board for each move for each row in the df
for i in range(len(df.iloc[:n_games_graph])):
    if df["ts_evals"][i][-1] > WIN_CUTOFF:
        result = "w"
    elif df["ts_evals"][i][-1] > -WIN_CUTOFF:
        result = "d"
    else:
        result = "l"
    alpha = 1 if result != "d" else 0.5
    color = {"w": "green", "l": "red", "d": "grey"}[result]

    plt.scatter(np.arange(len(df["ts_evals"][i])), df["ts_evals"][i], s=5, alpha=alpha, color=color)
    plt.plot(
        np.arange(len(df["ts_evals"][i])),
        df["ts_evals"][i],
        linewidth=0.3,
        alpha=alpha,
        color=color,
    )
plt.gca().axhspan(-WIN_CUTOFF, WIN_CUTOFF, facecolor="black", alpha=0.1)
plt.show()

# %%

# Confusing to interpret
# plt.title("Valid Matches")
## df[["white", "result"]].plot(kind="hist", by="white")
## plt.show()
# plt.hist(df.loc[df["white"] == "stockfish", "result"], label="stockfish", alpha=0.6)
# plt.hist(df.loc[df["white"] != "stockfish", "result"], label="gpt-3.5-turbo-instruct", alpha=0.6)
# plt.legend()
# plt.show()
