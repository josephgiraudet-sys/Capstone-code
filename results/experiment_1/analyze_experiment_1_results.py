import pandas as pd
import matplotlib
import re
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results") / "experiment_1"
BASE_RESULTS_PATH = RESULTS_DIR / "all_model_evaluations.csv"
CYCLER_RESULTS_PATH = Path("results/extra_cycler/cycler_model_evaluations.csv")

MAX_PAYOFFS = {
    "allc": 1000,
    "alld": 200,
    "cycler_ccd": 736,
    "cycler_ddc": 464,
    "generous_tft": 600,
    "grudger": 602,
    "hard_tft": 600,
    "joss": 540,
    "prober": 600,
    "random": 600,
    "soft_grudger": 600,
    "tft": 600,
    "tftt": 800,
}

SELF_PAYOFF_ORDER = [
    "allc",
    "alld",
    "tft",
    "hard_tft",
    "generous_tft",
    "grudger",
    "soft_grudger",
    "tftt",
    "prober",
    "joss",
    "random",
    "cycler_ccd",
    "cycler_ddc",
]


def load_results() -> pd.DataFrame:
    frames = [pd.read_csv(BASE_RESULTS_PATH)]

    if CYCLER_RESULTS_PATH.exists():
        frames.append(pd.read_csv(CYCLER_RESULTS_PATH))

    df = pd.concat(frames, ignore_index=True)
    return df[df["p_noise"] == 0.0].copy()


# Load results
df = load_results()
all_results_df = df.copy()


# --------------------------------------------------
# 1. CLASSIFY MODELS BY TRAINING OPPONENT
# --------------------------------------------------

def classify_model(name):
    match = re.match(r"^q_agent_vs_(.+)_seed\d+$", name)
    if match:
        return match.group(1)
    return name


def extract_trained_strategy(name):
    match = re.match(r"^q_agent_vs_(.+?)(?:_seed\d+)?$", name)
    if match:
        return match.group(1)
    return None


def format_strategy_name(name):
    display_names = {
        "allc": "ALLC",
        "alld": "ALLD",
        "cycler_ccd": "cycler CCD",
        "cycler_ddc": "cycler DDC",
        "tft": "TFT",
        "mist": "suspicious TFT",
        "hard_tft": "hard TFT",
        "tftt": "TFTT",
        "generous_tft": "generous TFT",
        "grudger": "grudger",
        "soft_grudger": "soft grudger",
        "joss": "Joss",
        "prober": "prober",
        "naive_prober": "naive prober",
        "random": "random",
    }
    return display_names.get(name, name.replace("_", " "))


def agent_label(strategy_name):
    return f"Agent trained against {format_strategy_name(strategy_name)}"


df["trained_strategy"] = df["model"].apply(classify_model)
all_results_df["trained_strategy"] = all_results_df["model"].apply(classify_model)

plot_groups = {
    "simple": ["allc", "alld"],
    "reciprocal_grudger": ["tft", "hard_tft", "generous_tft", "grudger", "soft_grudger"],
    "other": ["tftt", "prober", "joss", "random"],
}

model_order = [strategy for group in plot_groups.values() for strategy in group]
model_order = [strategy for strategy in model_order if strategy in set(df["trained_strategy"])]
opponent_order = sorted(df["test_opponent"].unique())

df = df[df["trained_strategy"].isin(model_order)].copy()
df["agent_label"] = df["trained_strategy"].apply(agent_label)


# --------------------------------------------------
# 2. TABLE: MEAN REWARD PER MODEL TYPE × OPPONENT
# --------------------------------------------------

table = (
    df.groupby(["trained_strategy", "test_opponent"])["mean_reward"]
    .mean()
    .reset_index()
)

pivot_table = table.pivot(index="trained_strategy", columns="test_opponent", values="mean_reward")
pivot_table = pivot_table.reindex(index=model_order, columns=opponent_order)
pivot_table.index = [agent_label(strategy) for strategy in pivot_table.index]

print("\n=== MEAN REWARD TABLE ===")
print(pivot_table)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pivot_table.to_csv(RESULTS_DIR / "mean_reward_table.csv")


# --------------------------------------------------
# 3. PLOTS: MEAN TOTAL REWARD VS OPPONENT
# --------------------------------------------------

def plot_reward_group(group_name, strategies, title, save_path, opponents=None):
    plt.figure(figsize=(18, 8))
    plot_opponents = opponents if opponents is not None else opponent_order

    for strategy in strategies:
        subset = table[table["trained_strategy"] == strategy]
        if len(subset) == 0:
            continue
        subset = subset.set_index("test_opponent").reindex(plot_opponents).reset_index()

        plt.plot(
            subset["test_opponent"],
            subset["mean_reward"],
            marker="o",
            label=agent_label(strategy),
        )

    plt.title(title)
    plt.xlabel("Test opponent")
    plt.ylabel("Mean total reward")
    plt.xticks(rotation=45, ha="right")
    plt.legend(ncol=2, fontsize=9)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


plot_reward_group(
    group_name="simple",
    strategies=plot_groups["simple"],
    title="Mean Total Reward: ALLC and ALLD Trained Agents",
    save_path=RESULTS_DIR / "plot_total_reward_allc_alld_agents.png",
)

plot_reward_group(
    group_name="reciprocal_grudger",
    strategies=["allc", "tft", "tftt", "generous_tft"],
    title="Mean Total Reward: ALLC, TFT, TFTT, and generous TFT Trained Agents",
    save_path=RESULTS_DIR / "plot_total_reward_tft_grudger_agents.png",
    opponents=["allc", "generous_tft", "tft", "tftt"],
)

plot_reward_group(
    group_name="other",
    strategies=plot_groups["other"],
    title="Mean Total Reward: TFTT, Prober, Joss, and Random Trained Agents",
    save_path=RESULTS_DIR / "plot_total_reward_other_agents.png",
)


# --------------------------------------------------
# 4. OVERALL MEAN REWARD PER MODEL
# --------------------------------------------------

overall = (
    df.groupby("trained_strategy")["mean_reward"]
    .mean()
    .reset_index()
)
overall["trained_strategy"] = pd.Categorical(overall["trained_strategy"], categories=model_order, ordered=True)
overall = overall.sort_values("trained_strategy")
overall["agent"] = overall["trained_strategy"].astype(str).apply(agent_label)

print("\n=== OVERALL MEAN REWARD PER MODEL ===")
print(overall)

overall.to_csv(RESULTS_DIR / "overall_mean_reward.csv", index=False)


# --------------------------------------------------
# 5. BAR PLOT: OVERALL PERFORMANCE
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.bar(overall["agent"], overall["mean_reward"])

plt.title("Overall Mean Total Reward per Agent")
plt.xlabel("Agent")
plt.ylabel("Mean total reward")
plt.xticks(rotation=45, ha="right")

plt.grid()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "plot_overall_mean_reward.png", bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 6. COOPERATION RATE AGAINST TRAINING STRATEGY
# --------------------------------------------------

coop = df[df["test_opponent"] == df["trained_strategy"]].copy()
coop = (
    coop.groupby("trained_strategy")["mean_cooperation_rate"]
    .mean()
    .reset_index()
)
coop["trained_strategy"] = pd.Categorical(coop["trained_strategy"], categories=model_order, ordered=True)
coop = coop.sort_values("trained_strategy")
coop["agent"] = coop["trained_strategy"].astype(str).apply(agent_label)

print("\n=== MEAN COOPERATION RATE AGAINST TRAINING STRATEGY ===")
print(coop)

plt.figure(figsize=(14, 6))

plt.bar(coop["agent"], coop["mean_cooperation_rate"])

plt.title("Cooperation Rate Against Training Strategy")
plt.xlabel("Agent")
plt.ylabel("Cooperation Rate")
plt.xticks(rotation=45, ha="right")

plt.grid()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "plot_cooperation_rate.png", bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 7. SELF-PLAY PERFORMANCE AS % OF MAX PAYOFF
# --------------------------------------------------

self_df = all_results_df.copy()
self_df["q_trained_strategy"] = self_df["model"].apply(extract_trained_strategy)
self_df = self_df[
    self_df["q_trained_strategy"].notna() &
    self_df["q_trained_strategy"].isin(MAX_PAYOFFS) &
    (self_df["test_opponent"] == self_df["q_trained_strategy"])
].copy()

self_summary = (
    self_df.groupby("q_trained_strategy", as_index=False)["mean_reward"]
    .mean()
)
self_summary["max_payoff"] = self_summary["q_trained_strategy"].map(MAX_PAYOFFS)
self_summary["payoff_pct_of_max"] = 100 * self_summary["mean_reward"] / self_summary["max_payoff"]

self_order = [
    strategy
    for strategy in SELF_PAYOFF_ORDER
    if strategy in set(self_summary["q_trained_strategy"])
]
self_summary["q_trained_strategy"] = pd.Categorical(
    self_summary["q_trained_strategy"],
    categories=self_order,
    ordered=True,
)
self_summary = self_summary.sort_values("q_trained_strategy")
self_summary["agent"] = self_summary["q_trained_strategy"].astype(str).apply(agent_label)

print("\n=== SELF-PLAY PAYOFF AS % OF MAX PAYOFF ===")
print(self_summary)

self_summary.to_csv(RESULTS_DIR / "self_strategy_payoff_pct_of_max.csv", index=False)

plt.figure(figsize=(14, 6))
plt.bar(self_summary["agent"], self_summary["payoff_pct_of_max"])
plt.title("Payoff Against Training Strategy as % of Max Payoff")
plt.xlabel("Agent")
plt.ylabel("Payoff (% of max attainable payoff)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 105)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot_self_strategy_payoff_pct_of_max.png", bbox_inches="tight")
plt.close()
