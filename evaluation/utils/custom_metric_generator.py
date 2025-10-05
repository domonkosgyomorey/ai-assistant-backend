import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_heatmap(df: pd.DataFrame, file_path: str):
    import os

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    pivot_data = (
        df.groupby(["metadata.question_type", "metadata.topic"])["correctness"].agg(["mean", "count"]).reset_index()
    )
    pivot_pass_rate = pivot_data.pivot(index="metadata.question_type", columns="metadata.topic", values="mean")
    pivot_count = pivot_data.pivot(index="metadata.question_type", columns="metadata.topic", values="count")

    fig, ax = plt.subplots(
        figsize=(max(14, len(pivot_pass_rate.columns) * 0.6), max(6, len(pivot_pass_rate.index) * 0.6))
    )

    sns.heatmap(
        pivot_pass_rate,
        annot=False,
        fmt=".0%",
        cmap="RdYlGn",
        center=0.8,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Pass Rate"},
        linewidths=0.5,
        ax=ax,
    )

    for i, _qtype in enumerate(pivot_pass_rate.index):
        for j, _topic in enumerate(pivot_pass_rate.columns):
            pass_rate = pivot_pass_rate.iloc[i, j]
            count = pivot_count.iloc[i, j]
            if not pd.isna(pass_rate):
                text_color = "white" if pass_rate < 0.5 else "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{pass_rate:.0%}\n(n={int(count)})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="bold",
                )
    ax.set_title("Pass Rate Heatmap: Question Type × Topic", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Topic", fontsize=12)
    ax.set_ylabel("Question Type", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_low_performance_bar_chart(metrics: pd.DataFrame, file_path: str):
    import os

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    low_perf = metrics[metrics["pass_rate"] < 0.7].sort_values(by="pass_rate").head(10)

    fig, ax = plt.subplots(figsize=(10, max(6, len(low_perf) * 0.5)))

    sns.barplot(
        data=low_perf,
        x="pass_rate",
        y="category",
        palette="Reds_r",
        ax=ax,
    )

    for i, (index, row) in enumerate(low_perf.iterrows()):
        ax.text(
            row["pass_rate"] + 0.01,
            i,
            f"{row['pass_rate']:.1%} (n={row['count']})",
            color="black",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Top 10 Lowest Performance Categories", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Pass Rate", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    plt.xlim(0, 1)
    plt.xticks(ticks=[i / 10 for i in range(11)], labels=[f"{i * 10}%" for i in range(11)])
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close()
