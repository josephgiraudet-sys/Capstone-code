from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results") / "experiment_5_bis"
RESULTS_PATH = RESULTS_DIR / "targeted_model_evaluations.csv"

OUTPUT_SELF = RESULTS_DIR / "targeted_self_strategy_diagnostics.csv"
OUTPUT_OVERALL = RESULTS_DIR / "targeted_overall_mean_reward.csv"

OUTPUT_PLOT_SELF_PAYOFF = RESULTS_DIR / "plot_targeted_self_payoff_pct_of_max.png"
OUTPUT_PLOT_OVERALL_MEAN_REWARD = RESULTS_DIR / "plot_targeted_overall_mean_reward.png"
OUTPUT_PLOT_COOPERATION_RATE = RESULTS_DIR / "plot_targeted_cooperation_rate.png"

STRATEGY_ORDER = ["hard_tft", "grudger"]
MODEL_ORDER = [
    "baseline",
    "improved_baseline",
    "gamma_099_lr_05_to_001_eps_09_to_0001",
    "gamma_0999_lr_05_to_001_eps_09_to_001",
]


def format_strategy_name(name: str) -> str:
    display_names = {
        "hard_tft": "hard TFT",
        "grudger": "grudger",
    }
    return display_names.get(name, name.replace("_", " "))


def format_model_name(name: str) -> str:
    display_names = {
        "baseline": "baseline",
        "improved_baseline": "improved baseline",
        "gamma_099_lr_05_to_001_eps_09_to_0001": "gamma .99, LR .5->.01",
        "gamma_0999_lr_05_to_001_eps_09_to_001": "gamma .999, LR .5->.01",
    }
    return display_names.get(name, name.replace("_", " "))


def add_model_variant(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model_variant" not in df.columns:
        df["model_variant"] = df["hyperparameter_config"].astype(str)
    if "initial_q_value_for_cooperate" not in df.columns:
        df["initial_q_value_for_cooperate"] = 0.0
    if "initial_q_value_for_defect" not in df.columns:
        df["initial_q_value_for_defect"] = 0.0

    baseline_mask = df["hyperparameter_config"].astype(str).eq("baseline")
    optimistic_q_mask = (
        df["initial_q_value_for_cooperate"].astype(float).eq(1.0)
        & df["initial_q_value_for_defect"].astype(float).eq(0.0)
    )
    df.loc[optimistic_q_mask & baseline_mask, "model_variant"] = "improved_baseline"
    return df


def ordered_model_variants(df: pd.DataFrame) -> list[str]:
    present = set(df["model_variant"].astype(str))
    ordered = [name for name in MODEL_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot_self_strategy_payoff(self_summary: pd.DataFrame, model_order: list[str]) -> None:
    x_positions = list(range(len(STRATEGY_ORDER)))
    bar_width = min(0.18, 0.75 / max(len(model_order), 1))

    plt.figure(figsize=(12, 7))
    for model_index, model_name in enumerate(model_order):
        values = []
        for strategy in STRATEGY_ORDER:
            row = self_summary[
                (self_summary["training_opponent"].astype(str) == strategy)
                & (self_summary["model_variant"].astype(str) == model_name)
            ]
            values.append(float(row["payoff_percent_of_max"].iloc[0]) if len(row) else 0.0)

        offsets = [
            x + (model_index - (len(model_order) - 1) / 2) * bar_width
            for x in x_positions
        ]
        plt.bar(offsets, values, width=bar_width, label=format_model_name(model_name))

    plt.title("Self-Strategy Payoff")
    plt.xlabel("Training opponent")
    plt.ylabel("Payoff (% of max attainable)")
    plt.xticks(x_positions, [format_strategy_name(strategy) for strategy in STRATEGY_ORDER])
    plt.ylim(0, 105)
    plt.grid(axis="y")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_SELF_PAYOFF, bbox_inches="tight")
    plt.close()


def plot_overall_mean_reward(overall: pd.DataFrame, model_order: list[str]) -> None:
    values = []
    labels = []
    for model_name in model_order:
        row = overall[overall["model_variant"].astype(str) == model_name]
        if len(row):
            values.append(float(row["mean_reward"].iloc[0]))
            labels.append(format_model_name(model_name))

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Overall Mean Reward")
    plt.xlabel("Model")
    plt.ylabel("Mean reward")
    plt.grid(axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_OVERALL_MEAN_REWARD, bbox_inches="tight")
    plt.close()


def plot_cooperation_rate(overall: pd.DataFrame, model_order: list[str]) -> None:
    values = []
    labels = []
    for model_name in model_order:
        row = overall[overall["model_variant"].astype(str) == model_name]
        if len(row):
            values.append(float(row["cooperation_rate"].iloc[0]))
            labels.append(format_model_name(model_name))

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Experiment 5 bis: Cooperation Rate")
    plt.xlabel("Model")
    plt.ylabel("Cooperation rate")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_COOPERATION_RATE, bbox_inches="tight")
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)
    df = add_model_variant(df)
    model_order = ordered_model_variants(df)

    overall = (
        df.groupby("model_variant", observed=False)[
            ["mean_reward", "payoff_percent_of_max", "cooperation_rate"]
        ]
        .mean()
        .reset_index()
    )
    overall["model_label"] = overall["model_variant"].astype(str).apply(format_model_name)
    overall.to_csv(OUTPUT_OVERALL, index=False)

    self_df = df[df["training_opponent"] == df["test_opponent"]].copy()
    self_summary = (
        self_df.groupby(["training_opponent", "model_variant"], observed=False)[
            ["mean_reward", "payoff_percent_of_max", "cooperation_rate"]
        ]
        .mean()
        .reset_index()
    )
    self_summary["model_label"] = self_summary["model_variant"].astype(str).apply(
        format_model_name
    )
    self_summary.to_csv(OUTPUT_SELF, index=False)

    print("\n=== OVERALL MEAN REWARD ===")
    print(overall)
    print("\n=== SELF-STRATEGY PAYOFF ===")
    print(self_summary)

    plot_self_strategy_payoff(self_summary, model_order)
    plot_overall_mean_reward(overall, model_order)
    plot_cooperation_rate(overall, model_order)


if __name__ == "__main__":
    main()
