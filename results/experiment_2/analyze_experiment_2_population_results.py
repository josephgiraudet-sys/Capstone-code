from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results") / "experiment_2"
RESULTS_PATH = RESULTS_DIR / "population_model_evaluations.csv"

OUTPUT_MEAN_REWARD_TABLE = RESULTS_DIR / "population_mean_reward_table.csv"
OUTPUT_OVERALL_MEAN_REWARD = RESULTS_DIR / "population_overall_mean_reward.csv"
OUTPUT_SELF_POPULATION_REWARD = RESULTS_DIR / "population_self_population_reward.csv"
OUTPUT_SELF_POPULATION_PAYOFF_PCT = (
    RESULTS_DIR / "population_self_population_payoff_pct_of_max.csv"
)

OUTPUT_PLOT_REWARD_BY_OPPONENT = RESULTS_DIR / "plot_population_total_reward_by_opponent.png"
OUTPUT_PLOT_OVERALL = RESULTS_DIR / "plot_population_overall_mean_reward.png"
OUTPUT_PLOT_SELF_POPULATION = RESULTS_DIR / "plot_population_self_population_reward.png"
OUTPUT_PLOT_SELF_POPULATION_PAYOFF_PCT = (
    RESULTS_DIR / "plot_population_self_population_payoff_pct_of_max.png"
)
OUTPUT_EXPLOITATIVE_TRANSFER_TABLE = (
    RESULTS_DIR / "population_exploitative_transfer_to_cooperative_opponents.csv"
)
OUTPUT_PLOT_EXPLOITATIVE_TRANSFER = (
    RESULTS_DIR / "plot_population_exploitative_transfer_to_cooperative_opponents.png"
)

MAX_ATTAINABLE_PAYOFFS = {
    "alld": 200,
    "generous_tft": 602,
    "hard_majo": 797,
    "hard_prober": 600,
    "hard_tft": 602,
    "joss": 602,
    "mist": 599,
    "prober": 602,
    "prober_3": 599,
    "soft_joss": 602,
    "soft_majo": 800,
    "tft": 602,
    "tftt": 802,
}

EXPLOITATIVE_TRANSFER_OPPONENTS = [
    "tft",
    "mist",
    "tftt",
    "generous_tft",
    "soft_majo",
]

EXPLOITATIVE_TRANSFER_LABELS = {
    "tft": "Tit for Tat",
    "mist": "Mistrust",
    "tftt": "Tit for Two Tats",
    "generous_tft": "Generous Tit for Tat",
    "soft_majo": "Soft Majority",
}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    return df[df["p_noise"] == 0.0].copy()


def classify_population_model(name: str) -> str:
    match = re.match(r"^q_agent_population_(.+)_seed\d+$", name)
    if match:
        return match.group(1)
    return name


def format_population_name(name: str) -> str:
    display_names = {
        "reciprocal": "Reciprocal",
        "exploitative": "Exploitative",
        "mixed": "Mixed",
    }
    return display_names.get(name, name.replace("_", " ").title())


def population_agent_label(population_name: str) -> str:
    return f"Agent trained on {format_population_name(population_name)} population"


def reward_plot_title(population_name: str) -> str:
    return f"Mean Total Reward: {format_population_name(population_name)} Population"


def format_strategy_name(name: str) -> str:
    display_names = {
        "alld": "ALLD",
        "generous_tft": "generous TFT",
        "hard_majo": "hard majority",
        "hard_prober": "hard prober",
        "hard_tft": "hard TFT",
        "joss": "Joss",
        "mist": "suspicious TFT",
        "prober": "prober",
        "prober_3": "prober 3",
        "soft_joss": "soft Joss",
        "soft_majo": "soft majority",
        "tft": "TFT",
        "tftt": "TFTT",
    }
    return display_names.get(name, name.replace("_", " "))


def plot_reward_group(
    table: pd.DataFrame,
    population_order: list[str],
    opponent_order: list[str],
) -> None:
    plt.figure(figsize=(16, 8))

    for population_name in population_order:
        subset = table[table["trained_population"] == population_name]
        subset = subset.set_index("test_opponent").reindex(opponent_order).reset_index()

        plt.plot(
            subset["test_opponent"],
            subset["payoff_pct_of_max"],
            marker="o",
            label=population_agent_label(population_name),
        )

    plt.title("Payoff as % of Max by Test Opponent for Population-Trained Agents")
    plt.xlabel("Test opponent")
    plt.ylabel("Payoff (% of max attainable)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_REWARD_BY_OPPONENT, bbox_inches="tight")
    plt.close()


def plot_exploitative_transfer(df: pd.DataFrame) -> None:
    subset = df[
        (df["trained_population"] == "exploitative")
        & (df["test_opponent"].isin(EXPLOITATIVE_TRANSFER_OPPONENTS))
    ].copy()
    if subset.empty:
        return

    subset["max_attainable_payoff"] = subset["test_opponent"].map(
        MAX_ATTAINABLE_PAYOFFS
    )
    subset["payoff_pct_of_max"] = (
        100 * subset["mean_reward"] / subset["max_attainable_payoff"]
    )

    transfer_summary = (
        subset.groupby("test_opponent")
        .agg(
            mean_reward=("mean_reward", "mean"),
            reward_std=("mean_reward", "std"),
            max_attainable_payoff=("max_attainable_payoff", "first"),
            payoff_pct_of_max=("payoff_pct_of_max", "mean"),
            payoff_pct_std=("payoff_pct_of_max", "std"),
        )
        .reset_index()
    )
    transfer_summary["test_opponent"] = pd.Categorical(
        transfer_summary["test_opponent"],
        categories=EXPLOITATIVE_TRANSFER_OPPONENTS,
        ordered=True,
    )
    transfer_summary = transfer_summary.sort_values("test_opponent")
    transfer_summary["test_opponent_label"] = transfer_summary[
        "test_opponent"
    ].map(EXPLOITATIVE_TRANSFER_LABELS)
    transfer_summary.to_csv(OUTPUT_EXPLOITATIVE_TRANSFER_TABLE, index=False)

    fig, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(
        transfer_summary["test_opponent_label"],
        transfer_summary["payoff_pct_of_max"],
        yerr=transfer_summary["payoff_pct_std"],
        capsize=5,
        color="#4C78A8",
        edgecolor="#2F4B64",
    )

    axis.set_title(
        "Exploitative-Trained Agent Transfer to Reciprocal and Cooperative Opponents"
    )
    axis.set_xlabel("Test opponent")
    axis.set_ylabel("Payoff (% of max attainable)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.35)
    axis.tick_params(axis="x", rotation=20)

    for bar, row in zip(bars, transfer_summary.itertuples(index=False)):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{row.mean_reward:.1f}\n({row.payoff_pct_of_max:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_EXPLOITATIVE_TRANSFER, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    all_results_df = df.copy()

    df["trained_population"] = df["model"].apply(classify_population_model)
    all_results_df["trained_population"] = all_results_df["model"].apply(
        classify_population_model
    )

    population_order = [
        population_name
        for population_name in ["reciprocal", "exploitative", "mixed"]
        if population_name in set(df["trained_population"])
    ]
    opponent_order = sorted(df["test_opponent"].unique())

    df = df[df["trained_population"].isin(population_order)].copy()
    df["agent_label"] = df["trained_population"].apply(population_agent_label)

    table = (
        df.groupby(["trained_population", "test_opponent"])["mean_reward"]
        .mean()
        .reset_index()
    )
    table["max_attainable_payoff"] = table["test_opponent"].map(
        MAX_ATTAINABLE_PAYOFFS
    )
    table["payoff_pct_of_max"] = (
        100 * table["mean_reward"] / table["max_attainable_payoff"]
    )

    pivot_table = table.pivot(
        index="trained_population",
        columns="test_opponent",
        values="mean_reward",
    )
    pivot_table = pivot_table.reindex(index=population_order, columns=opponent_order)
    pivot_table.index = [
        population_agent_label(population_name)
        for population_name in pivot_table.index
    ]

    print("\n=== POPULATION MEAN REWARD TABLE ===")
    print(pivot_table)
    pivot_table.to_csv(OUTPUT_MEAN_REWARD_TABLE)

    plot_reward_group(table, population_order, opponent_order)
    plot_exploitative_transfer(df)

    overall = (
        df.groupby("trained_population")["mean_reward"]
        .mean()
        .reset_index()
    )
    overall["trained_population"] = pd.Categorical(
        overall["trained_population"],
        categories=population_order,
        ordered=True,
    )
    overall = overall.sort_values("trained_population")
    overall["agent"] = overall["trained_population"].astype(str).apply(
        population_agent_label
    )

    print("\n=== POPULATION OVERALL MEAN REWARD ===")
    print(overall)
    overall.to_csv(OUTPUT_OVERALL_MEAN_REWARD, index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(overall["agent"], overall["mean_reward"])
    plt.title("Overall Mean Total Reward per Population-Trained Agent")
    plt.xlabel("Agent")
    plt.ylabel("Mean total reward")
    plt.xticks(rotation=20, ha="right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_OVERALL, bbox_inches="tight")
    plt.close()

    population_to_opponents = {
        "reciprocal": [
            "tft",
            "mist",
            "tftt",
            "hard_tft",
            "generous_tft",
        ],
        "exploitative": [
            "alld",
            "prober",
            "prober_3",
            "hard_prober",
        ],
        "mixed": [
            "tft",
            "hard_tft",
            "alld",
            "prober",
            "prober_3",
        ],
    }

    self_population_rows = []
    for population_name, opponents in population_to_opponents.items():
        subset = all_results_df[
            (all_results_df["trained_population"] == population_name)
            & (all_results_df["test_opponent"].isin(opponents))
        ].copy()

        if subset.empty:
            continue

        self_population_rows.append(
            {
                "trained_population": population_name,
                "agent": population_agent_label(population_name),
                "mean_reward": subset["mean_reward"].mean(),
                "mean_reward_per_round": subset["mean_reward_per_round"].mean(),
                "mean_cooperation_rate": subset["mean_cooperation_rate"].mean(),
            }
        )

    self_population_summary = pd.DataFrame(self_population_rows)
    if not self_population_summary.empty:
        self_population_summary["trained_population"] = pd.Categorical(
            self_population_summary["trained_population"],
            categories=population_order,
            ordered=True,
        )
        self_population_summary = self_population_summary.sort_values(
            "trained_population"
        )

    print("\n=== MEAN REWARD AGAINST OWN TRAINING POPULATION ===")
    print(self_population_summary)
    self_population_summary.to_csv(OUTPUT_SELF_POPULATION_REWARD, index=False)

    if not self_population_summary.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(
            self_population_summary["agent"],
            self_population_summary["mean_reward"],
        )
        plt.title("Mean Reward Against Own Training Population")
        plt.xlabel("Agent")
        plt.ylabel("Mean total reward")
        plt.xticks(rotation=20, ha="right")
        plt.grid()
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_SELF_POPULATION, bbox_inches="tight")
        plt.close()

    payoff_pct_rows = []
    for population_name, opponents in population_to_opponents.items():
        for opponent in opponents:
            subset = table[
                (table["trained_population"] == population_name)
                & (table["test_opponent"] == opponent)
            ]
            if subset.empty:
                continue

            max_payoff = MAX_ATTAINABLE_PAYOFFS[opponent]
            mean_reward = float(subset["mean_reward"].iloc[0])
            payoff_pct_rows.append(
                {
                    "trained_population": population_name,
                    "agent": population_agent_label(population_name),
                    "test_opponent": opponent,
                    "test_opponent_label": format_strategy_name(opponent),
                    "mean_reward": mean_reward,
                    "max_attainable_payoff": max_payoff,
                    "payoff_pct_of_max": 100 * mean_reward / max_payoff,
                }
            )

    payoff_pct = pd.DataFrame(payoff_pct_rows)

    print("\n=== PAYOFF AGAINST TRAINING POPULATION AS % OF MAX ATTAINABLE ===")
    print(payoff_pct)
    payoff_pct.to_csv(OUTPUT_SELF_POPULATION_PAYOFF_PCT, index=False)

    if not payoff_pct.empty:
        fig, axes = plt.subplots(
            nrows=len(population_order),
            ncols=1,
            figsize=(12, 4 * len(population_order)),
        )
        if len(population_order) == 1:
            axes = [axes]

        for axis, population_name in zip(axes, population_order):
            opponents = population_to_opponents[population_name]
            subset = payoff_pct[
                payoff_pct["trained_population"] == population_name
            ].copy()
            subset["test_opponent"] = pd.Categorical(
                subset["test_opponent"],
                categories=opponents,
                ordered=True,
            )
            subset = subset.sort_values("test_opponent")

            axis.bar(
                subset["test_opponent_label"],
                subset["payoff_pct_of_max"],
            )
            axis.set_title(
                f"{format_population_name(population_name)} Agent: "
                "Payoff Against Training Population"
            )
            axis.set_xlabel("Training-population opponent")
            axis.set_ylabel("Payoff (% of max attainable)")
            axis.set_ylim(0, 105)
            axis.grid(axis="y")
            axis.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_SELF_POPULATION_PAYOFF_PCT, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
