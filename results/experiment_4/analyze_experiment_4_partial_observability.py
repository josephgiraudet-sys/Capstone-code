from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results") / "experiment_4"
RESULTS_PATH = RESULTS_DIR / "partial_observability_model_evaluations.csv"

OUTPUT_OVERALL = RESULTS_DIR / "partial_observability_overall_mean_reward.csv"
OUTPUT_BY_TRAINING_OPPONENT = (
    RESULTS_DIR / "partial_observability_by_training_opponent.csv"
)
OUTPUT_SELF_PAYOFF_PCT = (
    RESULTS_DIR / "partial_observability_self_payoff_pct_of_max.csv"
)
OUTPUT_MEAN_REWARD_TABLE = RESULTS_DIR / "partial_observability_mean_reward_table.csv"

OUTPUT_PLOT_OVERALL = RESULTS_DIR / "plot_partial_observability_overall.png"
OUTPUT_PLOT_OVERALL_COOPERATION = (
    RESULTS_DIR / "plot_partial_observability_overall_cooperation_rate.png"
)
OUTPUT_PLOT_BY_TRAINING_OPPONENT = (
    RESULTS_DIR / "plot_partial_observability_by_training_opponent.png"
)
OUTPUT_PLOT_SELF_PAYOFF_PCT = (
    RESULTS_DIR / "plot_partial_observability_self_payoff_pct_of_max.png"
)

MAX_ATTAINABLE_PAYOFFS = {
    "cycler_ccd": 736,
    "cycler_ddc": 464,
    "hard_majo": 797,
    "hard_tft": 602,
    "soft_majo": 800,
    "tftt": 802,
}

STRATEGY_ORDER = [
    "tftt",
    "hard_tft",
    "cycler_ddc",
    "cycler_ccd",
    "hard_majo",
    "soft_majo",
]

HISTORY_ORDER = [1, 3]


def format_strategy_name(name: str) -> str:
    display_names = {
        "cycler_ccd": "Cycler CCD",
        "cycler_ddc": "Cycler DDC",
        "hard_majo": "hard majority",
        "hard_tft": "hard TFT",
        "soft_majo": "soft majority",
        "tftt": "TFTT",
    }
    return display_names.get(name, name.replace("_", " "))


def history_label(history_length: int) -> str:
    return f"last {history_length}"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)
    df["training_history_length"] = df["training_history_length"].astype(int)

    overall = (
        df.groupby("training_history_length")[
            [
                "mean_reward",
                "mean_reward_per_round",
                "mean_cooperation_rate",
                "mean_defection_rate",
            ]
        ]
        .mean()
        .reset_index()
    )
    overall["memory"] = overall["training_history_length"].apply(history_label)

    print("\n=== EXPERIMENT 4 OVERALL MEAN REWARD BY OBSERVATION LENGTH ===")
    print(overall)
    overall.to_csv(OUTPUT_OVERALL, index=False)

    plt.figure(figsize=(8, 6))
    plt.bar(overall["memory"], overall["mean_reward"])
    plt.title("Partial Observability: Overall Mean Reward")
    plt.xlabel("Observation history length")
    plt.ylabel("Mean total reward")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_OVERALL, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    cooperation_percent = 100 * overall["mean_cooperation_rate"]
    bars = plt.bar(
        overall["memory"],
        cooperation_percent,
        color=["#4C78A8", "#59A14F"],
        edgecolor="#2F4B64",
    )
    for bar, value in zip(bars, cooperation_percent):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title("Partial Observability: Overall Cooperation Rate")
    plt.xlabel("Observation history length")
    plt.ylabel("Mean cooperation rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_OVERALL_COOPERATION, bbox_inches="tight", dpi=300)
    plt.close()

    by_training_opponent = (
        df.groupby(["training_opponent", "training_history_length"])["mean_reward"]
        .mean()
        .reset_index()
    )
    by_training_opponent["training_opponent"] = pd.Categorical(
        by_training_opponent["training_opponent"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    by_training_opponent = by_training_opponent.sort_values(
        ["training_opponent", "training_history_length"]
    )
    by_training_opponent.to_csv(OUTPUT_BY_TRAINING_OPPONENT, index=False)

    print("\n=== MEAN REWARD BY TRAINING STRATEGY AND OBSERVATION LENGTH ===")
    print(by_training_opponent)

    x_positions = range(len(STRATEGY_ORDER))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    for index, history_length in enumerate(HISTORY_ORDER):
        subset = (
            by_training_opponent[
                by_training_opponent["training_history_length"] == history_length
            ]
            .set_index("training_opponent")
            .reindex(STRATEGY_ORDER)
            .reset_index()
        )
        offsets = [
            x + (index - (len(HISTORY_ORDER) - 1) / 2) * bar_width
            for x in x_positions
        ]
        plt.bar(
            offsets,
            subset["mean_reward"],
            width=bar_width,
            label=history_label(history_length),
        )

    plt.title("Partial Observability by Training Strategy")
    plt.xlabel("Training strategy")
    plt.ylabel("Mean total reward")
    plt.xticks(
        list(x_positions),
        [format_strategy_name(name) for name in STRATEGY_ORDER],
        rotation=30,
        ha="right",
    )
    plt.grid(axis="y")
    plt.legend(title="Observation")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_BY_TRAINING_OPPONENT, bbox_inches="tight")
    plt.close()

    mean_reward_table = (
        df.groupby(["training_opponent", "training_history_length", "test_opponent"])[
            "mean_reward"
        ]
        .mean()
        .reset_index()
    )
    mean_reward_pivot = mean_reward_table.pivot_table(
        index=["training_opponent", "training_history_length"],
        columns="test_opponent",
        values="mean_reward",
    )
    mean_reward_pivot = mean_reward_pivot.reindex(
        pd.MultiIndex.from_product(
            [STRATEGY_ORDER, HISTORY_ORDER],
            names=["training_opponent", "training_history_length"],
        )
    )
    mean_reward_pivot = mean_reward_pivot.reindex(columns=STRATEGY_ORDER)
    mean_reward_pivot.to_csv(OUTPUT_MEAN_REWARD_TABLE)

    self_df = df[df["training_opponent"] == df["test_opponent"]].copy()
    self_df["max_attainable_payoff"] = self_df["test_opponent"].map(
        MAX_ATTAINABLE_PAYOFFS
    )
    self_df["payoff_pct_of_max"] = (
        100 * self_df["mean_reward"] / self_df["max_attainable_payoff"]
    )

    self_summary = (
        self_df.groupby(["training_opponent", "training_history_length"])[
            ["mean_reward", "payoff_pct_of_max"]
        ]
        .mean()
        .reset_index()
    )
    self_summary["training_opponent"] = pd.Categorical(
        self_summary["training_opponent"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    self_summary = self_summary.sort_values(
        ["training_opponent", "training_history_length"]
    )
    self_summary.to_csv(OUTPUT_SELF_PAYOFF_PCT, index=False)

    print("\n=== SELF-OPPONENT PAYOFF AS % OF MAX ATTAINABLE ===")
    print(self_summary)

    plt.figure(figsize=(12, 6))
    for index, history_length in enumerate(HISTORY_ORDER):
        subset = (
            self_summary[
                self_summary["training_history_length"] == history_length
            ]
            .set_index("training_opponent")
            .reindex(STRATEGY_ORDER)
            .reset_index()
        )
        offsets = [
            x + (index - (len(HISTORY_ORDER) - 1) / 2) * bar_width
            for x in x_positions
        ]
        plt.bar(
            offsets,
            subset["payoff_pct_of_max"],
            width=bar_width,
            label=history_label(history_length),
        )

    plt.title("Self-Opponent Payoff Under Partial Observability")
    plt.xlabel("Training and test strategy")
    plt.ylabel("Payoff (% of max attainable)")
    plt.ylim(0, 105)
    plt.xticks(
        list(x_positions),
        [format_strategy_name(name) for name in STRATEGY_ORDER],
        rotation=30,
        ha="right",
    )
    plt.grid(axis="y")
    plt.legend(title="Observation")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_SELF_PAYOFF_PCT, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
