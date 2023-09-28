# %%
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


def try_ast_eval(s):
    try:
        return ast.literal_eval(s)
    except:
        return pd.NA


# trend towards AI playing better against better models
CSV_PATH = "results/results_2023_09_22_17_47_34_587661.csv"  # no trend
# CSV_PATH = "results/results_2023_09_22_18_01_40_144335.csv"


def decide_game(white=None, result=None, moves=None, eval_type=None, eval_value=None, **kwargs):
    """A numerical representation of the games result, according to stockfish's eval.
    positive is in favor of gpt, negative is in favor of stockfish
    """
    if pd.isna(eval_type):
        if result > 0.5:
            val = WIN_CP
        elif result < 0.5:
            val = -WIN_CP
        else:
            val = 0
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


def gpt_win_prob(gpt_cp_result=None, ply=None, **kwargs):
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

    value = gpt_cp_result  * 100 / NormalizeToPawnValue
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


def gpt_lose_prob(gpt_cp_result=None, ply=None, **kwargs):
    return gpt_win_prob(-gpt_cp_result, ply)


def draw_prob(gpt_cp_result, ply, **kwargs):
    w_p = gpt_win_prob(gpt_cp_result, ply)
    l_p = gpt_lose_prob(gpt_cp_result, ply)
    return 1 - w_p - l_p


def init_df(path):
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
    df["ply"] = df["moves"].apply(len)

    # Set positive numbers are better for gpt
    df["eval_value"].loc[df["white"] == "stockfish"] *= -1
    # df["result"].loc[df["white"] == "stockfish"] = 1 - df["result"].loc[df["white"] == "stockfish"]
    df.loc[df["white"] == "stockfish", "result"] = 1 - df["result"].loc[df["white"] == "stockfish"]

    df["gpt_cp_result"] = df.apply(lambda row: decide_game(**row), axis=1)
    # gpt_win_prob is in terms of white, but gpt_cp_result already converted to be in terms of GPT
    df["gpt_win_prob"] = df.apply(lambda row: gpt_win_prob(**row), axis=1)
    df["draw_prob"] = df.apply(lambda row: draw_prob(**row), axis=1)

    assert (
        df[df["result"] == 1].apply(
            lambda r: r["gpt_win_prob"] == 1 and r["gpt_cp_result"] == WIN_CP, axis=1
        )
    ).all()
    assert (
        df.query("result==0 and illegal_move.isna()").apply(
            lambda r: r["gpt_win_prob"] == 0 and r["gpt_cp_result"] == -WIN_CP, axis=1
        )
    ).all()
    assert (
        df.query('eval_type=="mate"')
        .apply(
            lambda r: (r["gpt_cp_result"] == WIN_CP and r["gpt_win_prob"] == 1)
            or (r["gpt_cp_result"] == -WIN_CP and r["gpt_win_prob"] == 0),
            axis=1,
        )
        .all()
    )
    ## Fails because with a very good stockfish, even a slight edge is a win, but I require 300 pawns worth of advantage
    # assert df.query("gpt_win_prob>0.99").apply(
    #        lambda r: (~r.isna()["illegal_move"] and r["gpt_cp_result"] >= WIN_CUTOFF)
    #        or r["result"] == 1,
    #        axis=1,
    # ).mean() > 0.9, "often GPT is predicted to win but determined GPT didn't/would not have won"
    assert (
        df.query("gpt_win_prob<0.01 and draw_prob < 0.01").apply(
            lambda r: r["result"] == 0
            or (r["gpt_cp_result"] <= -WIN_CUTOFF and ~r.isna()["illegal_move"]),
            axis=1,
        )
    ).all()

    _win_p = lambda cutoff: np.mean(
        [gpt_win_prob(gpt_cp_result=cutoff, ply=i) for i in np.arange(40, 80)]
    )
    if abs(_win_p(WIN_CUTOFF) - 0.5) >= 0.05:
        best_cutoff = min(np.arange(1, 800, 1), key=lambda c: abs(_win_p(c) - 0.5))
        print(f"picked a bad WIN_CUTOFF, for a cutoff with 50% chance of winning use {best_cutoff}")

    return df


if __name__ == "__main__":
    df = init_df(CSV_PATH)

    # df[df["result"] == 1][["gpt_win_prob", "gpt_cp_result"]]
    # %%

    # I think function is using the interval version, and returning the value not normalized by pawns?
    # WIN_CUTOFF=328 works as expected, =100 does not give +50% win chance

    def f(r):
        if r >= WIN_CUTOFF:
            return "win"
        elif r <= -WIN_CUTOFF:
            return "loss"
        else:
            return "draw"

    print(df.groupby(df.apply(lambda r: f(r["gpt_cp_result"]), axis=1))["gpt_win_prob"].mean())
    print(df.groupby(df.apply(lambda r: f(r["gpt_cp_result"]), axis=1))["draw_prob"].mean())

    plt.plot(
        np.arange(240),
        [gpt_win_prob(gpt_cp_result=500 * 3, ply=i) for i in np.arange(240)],
        label="win",
    )
    plt.plot(
        np.arange(240),
        [draw_prob(gpt_cp_result=WIN_CUTOFF, ply=i) for i in np.arange(240)],
        label="draw",
    )
    plt.plot(
        np.arange(240),
        [gpt_lose_prob(gpt_cp_result=WIN_CUTOFF, ply=i) for i in np.arange(240)],
        label="lose",
    )
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim(0, 240)
    # %%

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
    plt.title("Number of Forced mates for GPT after game ended by invalid move")
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
        (df["eval_type"] == "cp")
        & (df["white_elo"] >= elo_bins[0])
        & (df["white_elo"] < elo_bins[1]),
        "gpt_cp_result",
    ]

    for ax, (min_elo, max_elo) in zip(
        axs, zip_longest(elo_bins[:-1], elo_bins[1:], fillvalue=mx + 1)
    ):
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
    y = df["ply"].values
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
    # %% Plot Win Probability vs elo
    x = df["white_elo"].values
    y = df["gpt_cp_result"].values
    sns.regplot(x=x, y=y, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2})
    plt.title("Win probability vs Stockfish Elo")
    plt.xlabel("Elo")
    plt.ylabel("Win probability")
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
    df["ts_evals"] = df.iloc[:n_games_graph].apply(
        lambda r: get_evals(r["white"], r["moves"]), axis=1
    )

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

        plt.scatter(
            np.arange(len(df["ts_evals"][i])), df["ts_evals"][i], s=5, alpha=alpha, color=color
        )
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
