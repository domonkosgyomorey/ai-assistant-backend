import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from judger.judger_service import judger_service
from pydantic import BaseModel
from utils.common_types import JudgedTestCase
from utils.safe_utils import safe_open

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def create_output_directory(base_path: Path) -> Path:
    """Create and return the results directory path."""
    results_dir = base_path.parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def aggregate_by_question_type(
    test_cases: list[JudgedTestCase],
) -> dict[str, dict[str, BaseModel]]:
    """Aggregate judgements by question type."""
    group_by_question_type: dict[str, list[JudgedTestCase]] = defaultdict(list)
    for case in test_cases:
        question_type = case.question_type
        group_by_question_type[question_type].append(case)

    result: dict[str, dict[str, BaseModel]] = {}
    for question_type, cases in group_by_question_type.items():
        group_by_judgement: dict[str, list[BaseModel]] = {}
        for case in cases:
            for name, metrics in case.judgement.items():
                if name not in group_by_judgement:
                    group_by_judgement[name] = []
                group_by_judgement[name].append(metrics)

        for name, metrics_list in group_by_judgement.items():
            aggregated_metrics = judger_service.aggregate(metrics_list)
            if aggregated_metrics:
                if question_type not in result:
                    result[question_type] = {}
                result[question_type][name] = aggregated_metrics

    return result


def aggregate_by_group(test_cases: list[JudgedTestCase]) -> dict[str, dict[str, BaseModel]]:
    """Aggregate judgements by group."""
    group_by_group: dict[str, list[JudgedTestCase]] = defaultdict(list)
    for case in test_cases:
        group = case.groups
        if not group:
            group = ["default"]
        group_by_group[group[-1]].append(case)

    result: dict[str, dict[str, BaseModel]] = {}
    for group, cases in group_by_group.items():
        group_by_judgement: dict[str, list[BaseModel]] = {}
        for case in cases:
            for name, metrics in case.judgement.items():
                if name not in group_by_judgement:
                    group_by_judgement[name] = []
                group_by_judgement[name].append(metrics)

        for name, metrics_list in group_by_judgement.items():
            aggregated_metrics = judger_service.aggregate(metrics_list)
            if aggregated_metrics:
                if group not in result:
                    result[group] = {}
                result[group][name] = aggregated_metrics

    return result


def visualize_correctness_metrics(
    aggregated_data: dict[str, dict[str, BaseModel]],
    output_dir: Path,
    group_name: str = "question_type",
):
    """Create bar charts for correctness judger metrics."""
    correctness_data = {}
    for category, judgers in aggregated_data.items():
        if "correctness" in judgers:
            correctness = judgers["correctness"]
            correctness_data[category] = {
                "factual_correctness": correctness.average_factual_correctness,
                "completeness": correctness.average_completeness / 5.0,  # Normalize to 0-1
                "extra_info": correctness.average_extra_information,
            }

    if not correctness_data:
        return

    categories = list(correctness_data.keys())
    metrics = ["factual_correctness", "completeness", "extra_info"]
    metric_labels = ["Factual Correctness", "Completeness (normalized)", "Extra Information Rate"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Correctness Metrics by {group_name.replace('_', ' ').title()}", fontsize=16, fontweight="bold")

    colors = sns.color_palette("husl", len(categories))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels, strict=False)):
        ax = axes[idx]
        values = [correctness_data[cat][metric] for cat in categories]

        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xlabel(group_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    output_path = output_dir / f"correctness_by_{group_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_correctness_radar(
    aggregated_data: dict[str, dict[str, BaseModel]],
    output_dir: Path,
    group_name: str = "question_type",
):
    """Create radar chart for correctness judger metrics."""
    correctness_data = {}
    for category, judgers in aggregated_data.items():
        if "correctness" in judgers:
            correctness = judgers["correctness"]
            correctness_data[category] = {
                "Factual\nCorrectness": correctness.average_factual_correctness,
                "Completeness": correctness.average_completeness / 5.0,  # Normalize to 0-1
                "Extra\nInformation": correctness.average_extra_information,
            }

    if not correctness_data:
        return

    categories = list(correctness_data.keys())
    metric_labels = list(next(iter(correctness_data.values())).keys())

    num_vars = len(metric_labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    fig.suptitle(
        f"Correctness Metrics Radar Chart by {group_name.replace('_', ' ').title()}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = sns.color_palette("husl", len(categories))

    for idx, category in enumerate(categories):
        values = [correctness_data[category][label] for label in metric_labels]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, "o-", linewidth=2, label=category, color=colors[idx], alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, alpha=0.7)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = output_dir / f"correctness_radar_{group_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_all_metrics_radar(
    aggregated_data: dict[str, dict[str, BaseModel]],
    output_dir: Path,
    group_name: str = "question_type",
):
    """Create comprehensive radar chart with all key metrics."""
    all_metrics_data = {}

    for category, judgers in aggregated_data.items():
        metrics = {}

        if "correctness" in judgers:
            correctness = judgers["correctness"]
            metrics["Factual\nCorrectness"] = correctness.average_factual_correctness
            metrics["Completeness"] = correctness.average_completeness / 5.0  # Normalize to 0-1
            metrics["Extra\nInfo"] = correctness.average_extra_information

        if "keyword" in judgers:
            keyword = judgers["keyword"]
            metrics["Required\nKeywords"] = keyword.average_required_keyword_correctness
            metrics["Blocked\nKeywords"] = keyword.average_block_keyword_correctness

        if "grounded" in judgers:
            grounded = judgers["grounded"]
            metrics["Grounding\nQuality"] = 1.0 - (grounded.average_hallucination / 5.0)

        if metrics:
            all_metrics_data[category] = metrics

    if not all_metrics_data:
        return

    categories = list(all_metrics_data.keys())

    metric_labels = list(next(iter(all_metrics_data.values())).keys())

    num_vars = len(metric_labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection="polar"))
    fig.suptitle(
        f"All Metrics Radar Chart by {group_name.replace('_', ' ').title()}", fontsize=26, fontweight="bold", y=0.98
    )

    colors = sns.color_palette("husl", len(categories))

    for idx, category in enumerate(categories):
        values = [all_metrics_data[category].get(label, 0) for label in metric_labels]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, "o-", linewidth=3.5, label=category, color=colors[idx], alpha=0.85, markersize=12)
        ax.fill(angles, values, alpha=0.2, color=colors[idx])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=18, fontweight="bold")

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=16, alpha=0.8, fontweight="bold")

    from matplotlib.font_manager import FontProperties

    legend_font = FontProperties(size=16, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), prop=legend_font, framealpha=0.9)

    ax.grid(True, linestyle="--", alpha=0.5, linewidth=1.2)

    plt.tight_layout()
    output_path = output_dir / f"all_metrics_radar_{group_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_correctness_individual(
    aggregated_data: dict[str, dict[str, BaseModel]],
    output_dir: Path,
    group_name: str = "question_type",
):
    """Create individual bar charts for each correctness metric."""
    correctness_data = {}
    for category, judgers in aggregated_data.items():
        if "correctness" in judgers:
            correctness = judgers["correctness"]
            correctness_data[category] = {
                "factual_correctness": correctness.average_factual_correctness,
                "completeness": correctness.average_completeness / 5.0,  # Normalize to 0-1
                "extra_info": correctness.average_extra_information,
            }

    if not correctness_data:
        return

    categories = list(correctness_data.keys())
    metrics_info = [
        ("factual_correctness", "Factual Correctness", "#2ecc71"),
        ("completeness", "Completeness (Normalized)", "#3498db"),
        ("extra_info", "Extra Information Rate", "#e74c3c"),
    ]

    for metric_key, metric_title, color in metrics_info:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"{metric_title} by {group_name.replace('_', ' ').title()}", fontsize=16, fontweight="bold")

        values = [correctness_data[cat][metric_key] for cat in categories]
        bars = ax.bar(categories, values, color=color, alpha=0.8, edgecolor="black", linewidth=1.5)

        ax.set_ylabel("Score", fontsize=13, fontweight="bold")
        ax.set_xlabel(group_name.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        output_path = output_dir / f"correctness_{metric_key}_{group_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()


def visualize_keyword_metrics(
    aggregated_data: dict[str, dict[str, BaseModel]],
    output_dir: Path,
    group_name: str = "question_type",
):
    """Create bar charts for keyword judger metrics."""
    keyword_data = {}
    for category, judgers in aggregated_data.items():
        if "keyword" in judgers:
            keyword = judgers["keyword"]
            keyword_data[category] = {
                "required": keyword.average_required_keyword_correctness,
                "blocked": keyword.average_block_keyword_correctness,
            }

    if not keyword_data:
        return

    categories = list(keyword_data.keys())
    required_scores = [keyword_data[cat]["required"] for cat in categories]
    blocked_scores = [keyword_data[cat]["blocked"] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Keyword Metrics by {group_name.replace('_', ' ').title()}", fontsize=16, fontweight="bold")

    bars1 = ax.bar(x - width / 2, required_scores, width, label="Required Keywords", color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(x + width / 2, blocked_scores, width, label="Blocked Keywords", color="#e74c3c", alpha=0.8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel(group_name.replace("_", " ").title(), fontsize=12)
    ax.set_title("Keyword Correctness", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    output_path = output_dir / f"keyword_by_{group_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_grounded_metrics(
    aggregated_data: dict[str, dict[str, BaseModel]],
    output_dir: Path,
    group_name: str = "question_type",
):
    """Create bar chart for grounded judger metrics (hallucination)."""
    grounded_data = {}
    for category, judgers in aggregated_data.items():
        if "grounded" in judgers:
            grounded = judgers["grounded"]
            grounded_data[category] = 1.0 - (grounded.average_hallucination / 5.0)

    if not grounded_data:
        return

    categories = list(grounded_data.keys())
    values = [grounded_data[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Grounding Quality by {group_name.replace('_', ' ').title()}", fontsize=16, fontweight="bold")

    colors = sns.color_palette("RdYlGn", len(categories))
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Grounding Quality Score (1 - normalized hallucination)", fontsize=12)
    ax.set_xlabel(group_name.replace("_", " ").title(), fontsize=12)
    ax.set_title("Higher is Better (Less Hallucination)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = output_dir / f"grounded_by_{group_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_visualization(
    aggregated_data: dict[str, dict[str, BaseModel]],
    output_dir: Path,
    group_name: str = "question_type",
):
    """Create a comprehensive summary heatmap of all metrics."""
    categories = list(aggregated_data.keys())
    metric_names = []
    data_matrix = []

    for category in categories:
        row_data = []
        if not metric_names:  # First iteration, build metric names
            if "correctness" in aggregated_data[category]:
                metric_names.extend(["Factual Correct.", "Completeness", "Extra Info"])
            if "keyword" in aggregated_data[category]:
                metric_names.extend(["Required KW", "Blocked KW"])
            if "grounded" in aggregated_data[category]:
                metric_names.append("Grounding")

        if "correctness" in aggregated_data[category]:
            correctness = aggregated_data[category]["correctness"]
            row_data.extend(
                [
                    correctness.average_factual_correctness,
                    correctness.average_completeness / 5.0,
                    correctness.average_extra_information,
                ]
            )

        if "keyword" in aggregated_data[category]:
            keyword = aggregated_data[category]["keyword"]
            row_data.extend([keyword.average_required_keyword_correctness, keyword.average_block_keyword_correctness])

        if "grounded" in aggregated_data[category]:
            grounded = aggregated_data[category]["grounded"]
            row_data.append(1.0 - (grounded.average_hallucination / 5.0))

        data_matrix.append(row_data)

    if not data_matrix:
        return

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(f"All Metrics Summary by {group_name.replace('_', ' ').title()}", fontsize=24, fontweight="bold")

    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=metric_names,
        yticklabels=categories,
        cbar_kws={"label": "Score"},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        annot_kws={"fontsize": 16, "fontweight": "bold"},
    )

    ax.set_xlabel("Metrics", fontsize=18, fontweight="bold")
    ax.set_ylabel(group_name.replace("_", " ").title(), fontsize=18, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Score", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    output_path = output_dir / f"summary_heatmap_{group_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from judged test cases.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the judged test case JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    print(f"📊 Loading judged test cases from: {input_path}")
    with safe_open(input_path, mode="r", encoding="utf-8") as f:
        test_cases: list[JudgedTestCase] = [JudgedTestCase(**case) for case in json.load(f)]

    print(f"✓ Loaded {len(test_cases)} test cases\n")

    results_dir = create_output_directory(input_path)
    print(f"📁 Results will be saved to: {results_dir}\n")

    print("📈 Aggregating metrics by question type...")
    by_question_type = aggregate_by_question_type(test_cases)

    print("📈 Aggregating metrics by group...")
    by_group = aggregate_by_group(test_cases)

    if by_question_type:
        print("\n🎨 Generating visualizations by question type...")
        visualize_all_metrics_radar(by_question_type, results_dir, "question_type")
        create_summary_visualization(by_question_type, results_dir, "question_type")

    if by_group and len(by_group) > 1:  # Only if there are multiple groups
        print("\n🎨 Generating visualizations by group...")
        visualize_all_metrics_radar(by_group, results_dir, "group")
        create_summary_visualization(by_group, results_dir, "group")

    print("\n💾 Saving aggregated metrics as JSON...")
    json_output = {
        "by_question_type": {
            category: {judger: metrics.model_dump() for judger, metrics in judgers.items()}
            for category, judgers in by_question_type.items()
        },
        "by_group": {
            category: {judger: metrics.model_dump() for judger, metrics in judgers.items()}
            for category, judgers in by_group.items()
        },
    }

    json_path = results_dir / "aggregated_metrics.json"
    with safe_open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"✓ Saved: {json_path}")

    print("\n✅ Visualization complete! Check the results folder for generated charts.")
