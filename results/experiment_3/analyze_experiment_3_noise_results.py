from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results") / "experiment_3"
RESULTS_PATH = RESULTS_DIR / "population_noise_model_evaluations.csv"

OUTPUT_OVERALL = RESULTS_DIR / "population_noise_overall_mean_reward.csv"
OUTPUT_MATRIX = RESULTS_DIR / "population_noise_train_eval_matrix.csv"
OUTPUT_BY_POPULATION = RESULTS_DIR / "population_noise_by_population.csv"
OUTPUT_BY_TRAINING_NOISE = RESULTS_DIR / "population_noise_by_training_noise.csv"

OUTPUT_PLOT_MATRIX = RESULTS_DIR / "plot_population_noise_train_eval_matrix.png"
OUTPUT_PLOT_BY_POPULATION = RESULTS_DIR / "plot_population_noise_by_population.png"
OUTPUT_PLOT_BY_TRAINING_NOISE = (
    RESULTS_DIR / "plot_population_noise_by_training_noise.png"
)
OUTPUT_PLOT_HIGH_NOISE_DROP = (
    RESULTS_DIR / "plot_population_noise_high_noise_drop.png"
)
def format_population_name(name: str) -> str:
    display_names = {
        "reciprocal": "Reciprocal",
        "exploitative": "Exploitative",
        "mixed": "Mixed",
    }
    return display_names.get(name, name.replace("_", " ").title())


def population_order() -> list[str]:
    return ["exploitative", "mixed", "reciprocal"]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)
    df["training_p_noise"] = df["training_p_noise"].astype(float)
    df["p_noise"] = df["p_noise"].astype(float)

    overall = (
        df.groupby(["training_population", "training_p_noise", "p_noise"])[
            [
                "mean_reward",
                "mean_reward_per_round",
                "mean_cooperation_rate",
                "mean_defection_rate",
            ]
        ]
        .mean()
        .reset_index()
        .rename(columns={"p_noise": "eval_p_noise"})
    )

    print("\n=== NOISY POPULATION OVERALL MEAN REWARD ===")
    print(overall)
    overall.to_csv(OUTPUT_OVERALL, index=False)

    matrix = (
        df.groupby(["training_p_noise", "p_noise"])["mean_reward"]
        .mean()
        .reset_index()
        .pivot(
            index="training_p_noise",
            columns="p_noise",
            values="mean_reward",
        )
    )
    matrix.index.name = "training_p_noise"
    matrix.columns.name = "eval_p_noise"

    print("\n=== TRAIN NOISE BY EVAL NOISE MATRIX ===")
    print(matrix)
    matrix.to_csv(OUTPUT_MATRIX)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix.values, aspect="auto")
    plt.colorbar(label="Mean total reward")
    plt.xticks(range(len(matrix.columns)), matrix.columns)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.xlabel("Evaluation noise")
    plt.ylabel("Training noise")
    plt.title("Population Training Robustness Under Noise")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_MATRIX, bbox_inches="tight")
    plt.close()

    by_population = (
        df.groupby(["training_population", "training_p_noise", "p_noise"])[
            "mean_reward"
        ]
        .mean()
        .reset_index()
        .rename(columns={"p_noise": "eval_p_noise"})
    )
    by_population.to_csv(OUTPUT_BY_POPULATION, index=False)

    by_training_noise = (
        df.groupby(["training_population", "training_p_noise"])["mean_reward"]
        .mean()
        .reset_index()
    )
    by_training_noise["training_population"] = pd.Categorical(
        by_training_noise["training_population"],
        categories=population_order(),
        ordered=True,
    )
    by_training_noise = by_training_noise.sort_values(
        ["training_population", "training_p_noise"]
    )
    by_training_noise.to_csv(OUTPUT_BY_TRAINING_NOISE, index=False)

    training_noise_levels = sorted(by_training_noise["training_p_noise"].unique())
    x_positions = range(len(population_order()))
    bar_width = 0.18

    plt.figure(figsize=(11, 6))
    for index, training_noise in enumerate(training_noise_levels):
        subset = (
            by_training_noise[
                by_training_noise["training_p_noise"] == training_noise
            ]
            .set_index("training_population")
            .reindex(population_order())
            .reset_index()
        )
        offsets = [
            x + (index - (len(training_noise_levels) - 1) / 2) * bar_width
            for x in x_positions
        ]
        plt.bar(
            offsets,
            subset["mean_reward"],
            width=bar_width,
            label=f"train noise={training_noise:g}",
        )

    plt.title("Mean Reward by Population and Training Noise")
    plt.xlabel("Training population")
    plt.ylabel("Mean total reward")
    plt.xticks(
        list(x_positions),
        [format_population_name(name) for name in population_order()],
    )
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_BY_TRAINING_NOISE, bbox_inches="tight")
    plt.close()

    high_noise_drop_rows = []
    for population_name in population_order():
        subset = by_training_noise[
            by_training_noise["training_population"] == population_name
        ]
        clean_reward = float(
            subset[subset["training_p_noise"] == 0.0]["mean_reward"].iloc[0]
        )
        high_noise_reward = float(
            subset[subset["training_p_noise"] == 0.2]["mean_reward"].iloc[0]
        )
        high_noise_drop_rows.append(
            {
                "training_population": population_name,
                "clean_training_mean_reward": clean_reward,
                "high_noise_training_mean_reward": high_noise_reward,
                "reward_drop": clean_reward - high_noise_reward,
            }
        )

    high_noise_drop = pd.DataFrame(high_noise_drop_rows)

    plt.figure(figsize=(9, 6))
    plt.bar(
        [
            format_population_name(name)
            for name in high_noise_drop["training_population"]
        ],
        high_noise_drop["reward_drop"],
    )
    plt.title("Performance Drop from No-Noise to High-Noise Training")
    plt.xlabel("Training population")
    plt.ylabel("Mean reward drop: train 0.00 minus train 0.20")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_HIGH_NOISE_DROP, bbox_inches="tight")
    plt.close()

    clean_eval_average = (
        by_population[by_population["eval_p_noise"] == 0.0]
        .groupby("training_p_noise")["mean_reward"]
        .mean()
        .reset_index()
        .sort_values("training_p_noise")
    )

    plt.figure(figsize=(9, 6))
    plt.plot(
        clean_eval_average["training_p_noise"],
        clean_eval_average["mean_reward"],
        marker="o",
        linewidth=2,
        color="#4C78A8",
    )

    for row in clean_eval_average.itertuples(index=False):
        plt.text(
            row.training_p_noise,
            row.mean_reward + 4,
            f"{row.mean_reward:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title("Clean Evaluation Performance After Noisy Population Training")
    plt.xlabel("Training noise")
    plt.ylabel("Mean total reward")
    plt.ylim(330, 450)
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_BY_POPULATION, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
