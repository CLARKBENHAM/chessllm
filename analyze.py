# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/results_2023_09_21_21_48_55_335214.csv")

# evals NA if no illegal move
assert (df["illegal_move"].isna() == df["eval"].isna()).all()
df["eval_type"] = df["eval"].apply(lambda e: eval(e)["type"] if isinstance(e, str) else pd.NA)
df["eval_value"] = (
    df["eval"].apply(lambda e: eval(e)["value"] if isinstance(e, str) else pd.NA).astype("Int64")
)

# df[df["eval_type"].isna(), "eval_type"]= "mate"
# df[df["eval_value"].isna(), "eval_value"]= df[df["eval_value"].isna(), "result"]

# Set positive numbers are better for gpt
df["eval_value"].loc[df["white"] == "stockfish"] *= -1
df["result"].loc[df["white"] == "stockfish"] = 1 - df["result"].loc[df["white"] == "stockfish"]
df[df["eval_type"] == "mate"]
# %%
plt.title("Valid Matches")
# df[["white", "result"]].plot(kind="hist", by="white")
# plt.show()
plt.hist(df.loc[df["white"] == "stockfish", "result"], label="stockfish", alpha=0.6)
plt.hist(df.loc[df["white"] != "stockfish", "result"], label="gpt-3.5-turbo-instruct", alpha=0.6)
plt.legend()
plt.show()
# %%
plt.title("Evaluation of invalid moves")
plt.hist(df.loc[df["eval_type"] == "cp", "eval_value"], bins=30)
plt.show()

plt.title("Forced mates after invalid moves")
plt.hist(df.loc[df["eval_type"] == "mate", "eval_value"], bins=30)
plt.show()
# %%
