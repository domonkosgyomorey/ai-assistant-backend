"""
Visualize and compare metrics from UMAP vs PCA+UMAP pipelines on the same plots.

Usage:
    python visualize_pipeline_comparison.py --base_dir pipeline_comparison --output_dir comparison_plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(filepath: str) -> list[dict]:
    """Load metrics from JSON file."""
    if not Path(filepath).exists():
        return []

    with open(filepath, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    return data


def plot_distortion_comparison(umap_dir: str, pca_umap_dir: str, output_file: str) -> None:
    """Create distortion metrics radar and small textual clustering comparison below it.

    The figure contains the radar (spider) showing the four distortion metrics for
    both pipelines, and a compact textual panel underneath that lists selected
    clustering metrics (silhouette, noise_ratio) with relative percent differences.
    """
    # Load distortion metrics
    umap_dist_metrics = load_metrics(f"{umap_dir}/distortion_metrics.json")
    pca_umap_dist_metrics = load_metrics(f"{pca_umap_dir}/distortion_metrics.json")

    if not umap_dist_metrics or not pca_umap_dist_metrics:
        print("⚠️  Missing distortion metrics")
        return

    umap_after = [m for m in umap_dist_metrics if m.get("stage") == "after_reduction"]
    pca_umap_after = [m for m in pca_umap_dist_metrics if m.get("stage") == "after_reduction"]

    if not umap_after or not pca_umap_after:
        print("⚠️  No after-reduction metrics found")
        return

    umap_dist = {
        "topology": np.mean([m.get("trustworthiness", 0) for m in umap_after]),
        "geo_distance": np.mean([m.get("geodesic_preservation", 0) for m in umap_after]),
        "density": np.mean([m.get("density_preservation", 0) for m in umap_after]),
        "knn": np.mean([m.get("knn_preservation", 0) for m in umap_after]),
    }

    pca_umap_dist = {
        "topology": np.mean([m.get("trustworthiness", 0) for m in pca_umap_after]),
        "geo_distance": np.mean([m.get("geodesic_preservation", 0) for m in pca_umap_after]),
        "density": np.mean([m.get("density_preservation", 0) for m in pca_umap_after]),
        "knn": np.mean([m.get("knn_preservation", 0) for m in pca_umap_after]),
    }

    # Load clustering metrics (latest)
    umap_cluster_metrics = load_metrics(f"{umap_dir}/clustering_metrics.json")
    pca_umap_cluster_metrics = load_metrics(f"{pca_umap_dir}/clustering_metrics.json")

    umap_cluster = umap_cluster_metrics[-1] if umap_cluster_metrics else {}
    pca_umap_cluster = pca_umap_cluster_metrics[-1] if pca_umap_cluster_metrics else {}

    # Selected clustering metrics to display
    clustering_keys = [
        ("silhouette_score", True, "Silhouette"),  # (key, higher_is_better, label)
        ("noise_ratio", False, "Noise Ratio"),
    ]

    # Build figure: radar on top, text panel below - more compact, no title
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25)

    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    _plot_radar_on_axis(ax_radar, umap_dist, pca_umap_dist)

    # Text panel
    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis("off")

    # Prepare lines
    lines = []
    lines.append("Clustering metrics comparison (latest run):")
    for key, higher_better, label in clustering_keys:
        u_val = umap_cluster.get(key, None)
        p_val = pca_umap_cluster.get(key, None)
        if u_val is None and p_val is None:
            continue

        # Safe numeric conversion
        try:
            u = float(u_val) if u_val is not None else float("nan")
        except Exception:
            u = float("nan")
        try:
            p = float(p_val) if p_val is not None else float("nan")
        except Exception:
            p = float("nan")

        # Compute percent difference relative to average magnitude to avoid divide-by-zero
        denom = abs(p) if abs(p) > 1e-8 else (abs(u) if abs(u) > 1e-8 else 1.0)
        pct = (u - p) / denom * 100.0

        # Interpret which is better
        if np.isnan(u) or np.isnan(p):
            cmp_text = f"{label}: data missing"
        else:
            if higher_better:
                better = "UMAP" if u > p else "PCA+UMAP" if p > u else "equal"
                cmp_text = f"{label}: UMAP={u:.3f} | PCA+UMAP={p:.3f} → {better} by {abs(pct):.1f}%"
            else:
                # lower is better (e.g., noise_ratio, davies_bouldin)
                better = "UMAP" if u < p else "PCA+UMAP" if p < u else "equal"
                cmp_text = f"{label}: UMAP={u:.3f} | PCA+UMAP={p:.3f} → {better} by {abs(pct):.1f}%"

        lines.append(cmp_text)

    block_text = "\n".join(lines)

    # Draw multi-line text with larger monospace font for better readability
    ax_text.text(
        0.01,
        0.95,
        block_text,
        ha="left",
        va="top",
        fontsize=16,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.8", linewidth=2),
    )

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Distortion + clustering summary saved to {output_file}")


def plot_clustering_comparison(umap_dir: str, pca_umap_dir: str, output_file: str) -> None:
    """Create clustering metrics comparison with percentage differences."""
    umap_cluster_metrics = load_metrics(f"{umap_dir}/clustering_metrics.json")
    pca_umap_cluster_metrics = load_metrics(f"{pca_umap_dir}/clustering_metrics.json")

    if not umap_cluster_metrics or not pca_umap_cluster_metrics:
        print("⚠️  Missing clustering metrics")
        return

    # Get latest clustering metrics
    umap_cluster = umap_cluster_metrics[-1]
    pca_umap_cluster = pca_umap_cluster_metrics[-1]

    # Extract metrics
    umap_silhouette = umap_cluster.get("silhouette_score", 0) or 0
    pca_umap_silhouette = pca_umap_cluster.get("silhouette_score", 0) or 0
    umap_noise = umap_cluster.get("noise_ratio", 0)
    pca_umap_noise = pca_umap_cluster.get("noise_ratio", 0)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Clustering Quality: UMAP vs PCA+UMAP", fontsize=20, fontweight="bold")

    # 1. Silhouette Score (left)
    ax = axes[0]
    silhouette_values = [umap_silhouette, pca_umap_silhouette]
    bars = ax.bar(
        ["UMAP", "PCA+UMAP"], silhouette_values, color=["#3498db", "#e74c3c"], alpha=0.8, edgecolor="black", linewidth=2
    )

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    # Percentage difference
    if umap_silhouette != 0:
        pct_diff = ((pca_umap_silhouette - umap_silhouette) / abs(umap_silhouette)) * 100
        better = "PCA+UMAP" if pct_diff > 0 else "UMAP"
        color = "#27ae60" if pct_diff > 0 else "#c0392b"
        ax.text(
            0.5,
            max(silhouette_values) * 0.9,
            f"{better} is {abs(pct_diff):.1f}% better",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=color, linewidth=2),
            transform=ax.transData,
        )

    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title("Silhouette Score\n(higher is better)", fontsize=16, fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # 2. Noise Ratio (right)
    ax = axes[1]
    noise_values = [umap_noise, pca_umap_noise]
    bars = ax.bar(
        ["UMAP", "PCA+UMAP"], noise_values, color=["#3498db", "#e74c3c"], alpha=0.8, edgecolor="black", linewidth=2
    )

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    # Percentage difference (lower is better for noise)
    if umap_noise != 0:
        pct_diff = ((umap_noise - pca_umap_noise) / abs(umap_noise)) * 100
        better = "UMAP" if pct_diff < 0 else "PCA+UMAP"
        color = "#27ae60" if pct_diff > 0 else "#c0392b"
        ax.text(
            0.5,
            max(noise_values) * 0.9,
            f"{better} is {abs(pct_diff):.1f}% better",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=color, linewidth=2),
            transform=ax.transData,
        )

    ax.set_ylabel("Ratio", fontsize=14, fontweight="bold")
    ax.set_title("Noise Ratio\n(lower is better)", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(noise_values) * 1.3 if max(noise_values) > 0 else 1)
    ax.grid(axis="y", alpha=0.3)

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Clustering comparison saved to {output_file}")


def _plot_radar_on_axis(ax, umap_metrics: dict, pca_umap_metrics: dict) -> None:
    """Plot radar chart on given axis."""
    categories = ["Topology", "Geo Distance", "Density", "KNN Neighborhood"]

    umap_values = [
        umap_metrics["topology"],
        umap_metrics["geo_distance"],
        umap_metrics["density"],
        umap_metrics["knn"],
    ]

    pca_umap_values = [
        pca_umap_metrics["topology"],
        pca_umap_metrics["geo_distance"],
        pca_umap_metrics["density"],
        pca_umap_metrics["knn"],
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    umap_values += umap_values[:1]
    pca_umap_values += pca_umap_values[:1]
    angles += angles[:1]

    # Plot both pipelines
    ax.plot(angles, umap_values, "o-", linewidth=4, color="#3498db", markersize=14)
    ax.fill(angles, umap_values, alpha=0.15, color="#3498db")

    ax.plot(angles, pca_umap_values, "s-", linewidth=4, color="#e74c3c", markersize=14)
    ax.fill(angles, pca_umap_values, alpha=0.15, color="#e74c3c")

    # Reference line
    ax.plot(angles, [0.7] * len(angles), "--", color="green", alpha=0.5, linewidth=2.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=22, fontweight="bold")

    ax.set_ylim(0, 1.15)  # Extended to give more space for labels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=16)
    ax.grid(True, alpha=0.3)


def _plot_distortion_bars(ax, umap_metrics: dict, pca_umap_metrics: dict) -> None:
    """Plot distortion metrics as grouped bars."""
    metrics_names = ["Topology", "Geo\nDistance", "Density", "KNN\nNeighbor"]
    metrics_keys = ["topology", "geo_distance", "density", "knn"]

    x = np.arange(len(metrics_names))
    width = 0.35

    umap_vals = [umap_metrics[k] for k in metrics_keys]
    pca_umap_vals = [pca_umap_metrics[k] for k in metrics_keys]

    bars1 = ax.bar(x - width / 2, umap_vals, width, label="UMAP", color="#3498db", alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x + width / 2, pca_umap_vals, width, label="PCA+UMAP", color="#e74c3c", alpha=0.8, edgecolor="black")

    # Value labels
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

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Distortion Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(0.7, color="green", linestyle="--", linewidth=1.5, alpha=0.5, label="Good")
    ax.grid(axis="y", alpha=0.3)


def _plot_silhouette(ax, umap_cluster: dict, pca_umap_cluster: dict) -> None:
    """Plot silhouette score comparison."""
    silhouette = [umap_cluster.get("silhouette_score", 0) or 0, pca_umap_cluster.get("silhouette_score", 0) or 0]
    bars = ax.bar(
        ["UMAP", "PCA+UMAP"], silhouette, color=["#3498db", "#e74c3c"], alpha=0.8, edgecolor="black", linewidth=2
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Silhouette Score\n(higher is better)", fontsize=14, fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)


def _plot_n_clusters(ax, umap_cluster: dict, pca_umap_cluster: dict) -> None:
    """Plot number of clusters comparison."""
    n_clusters = [umap_cluster.get("n_clusters", 0), pca_umap_cluster.get("n_clusters", 0)]
    bars = ax.bar(
        ["UMAP", "PCA+UMAP"], n_clusters, color=["#3498db", "#e74c3c"], alpha=0.8, edgecolor="black", linewidth=2
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Number of Clusters", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _plot_noise_ratio(ax, umap_cluster: dict, pca_umap_cluster: dict) -> None:
    """Plot noise ratio comparison."""
    noise_ratio = [umap_cluster.get("noise_ratio", 0), pca_umap_cluster.get("noise_ratio", 0)]
    bars = ax.bar(
        ["UMAP", "PCA+UMAP"], noise_ratio, color=["#3498db", "#e74c3c"], alpha=0.8, edgecolor="black", linewidth=2
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Ratio", fontsize=12, fontweight="bold")
    ax.set_title("Noise Ratio\n(lower is better)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(noise_ratio) * 1.3 if max(noise_ratio) > 0 else 1)
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Compare UMAP vs PCA+UMAP pipeline metrics (2 separate plots)")
    parser.add_argument("--base_dir", required=True, help="Base directory with pipeline results")
    parser.add_argument("--output_dir", default="comparison_plots", help="Output directory for plots")

    args = parser.parse_args()

    umap_dir = f"{args.base_dir}/umap_pipeline"
    pca_umap_dir = f"{args.base_dir}/pca_umap_pipeline"

    # Check directories exist
    if not Path(umap_dir).exists() or not Path(pca_umap_dir).exists():
        print("❌ Pipeline directories not found:")
        print(f"   UMAP: {umap_dir}")
        print(f"   PCA+UMAP: {pca_umap_dir}")
        return

    print("📊 Generating comparison plots...")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: Distortion metrics (radar + bars)
    distortion_output = f"{args.output_dir}/1_distortion_comparison.png"
    plot_distortion_comparison(umap_dir, pca_umap_dir, distortion_output)

    # Plot 2: Clustering metrics (silhouette, clusters, noise)
    clustering_output = f"{args.output_dir}/2_clustering_comparison.png"
    plot_clustering_comparison(umap_dir, pca_umap_dir, clustering_output)

    print(f"\n✅ All comparison plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
