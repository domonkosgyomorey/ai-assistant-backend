import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "server"))

from core.config.config import config
from dal.mongo_db import MongoRetriever


def analyze_and_plot_similarity_scores(query: str, output_filename: str = "similarity_scores_analysis.png"):
    print(f"Elemzés indítása a lekérdezéssel: '{query}'")

    retriever = MongoRetriever(config.mongo.COLLECTION_NAME)

    print("Hasonlósági pontszámok lekérése...")
    results = retriever.get_all_similarity_scores(query)

    if not results:
        print("Nincsenek eredmények!")
        return

    scores = [result["similarity_score"] for result in results]
    scores_100 = [int(score * 100) for score in scores]

    print(f"Összesen {len(scores_100)} dokumentum található.")
    print(f"Pontszámok tartománya: {min(scores_100)}-{max(scores_100)}")

    scores_array = np.array(scores_100)
    percentile_90 = np.percentile(scores_array, 90)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)

    print(f"Átlag: {mean_score:.2f}")
    print(f"Szórás: {std_score:.2f}")
    print(f"90. percentilis: {percentile_90:.2f}")

    counter = Counter(scores_100)
    bins = list(range(0, 101))
    frequencies = [counter.get(i, 0) for i in bins]

    gradients = []
    for i in range(1, len(frequencies)):
        grad = abs(frequencies[i] - frequencies[i - 1])
        gradients.append((i, grad))

    high_range_gradients = [(i, grad) for i, grad in gradients if i >= 50]
    if high_range_gradients:
        sharpest_change = max(high_range_gradients, key=lambda x: x[1])
        change_point = sharpest_change[0]
    else:
        change_point = int(percentile_90)

    print(f"Legélesebb változás pontja: {change_point}")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.bar(bins, frequencies, alpha=0.7, color="skyblue", edgecolor="navy", width=0.8)
    plt.axvline(
        x=percentile_90, color="red", linestyle="--", linewidth=2, label=f"90. percentilis: {percentile_90:.0f}"
    )
    plt.axvline(x=change_point, color="orange", linestyle="--", linewidth=2, label=f"Éles változás: {change_point}")
    plt.axvline(x=mean_score, color="green", linestyle="-", linewidth=2, label=f"Átlag: {mean_score:.1f}")
    plt.xlabel("Hasonlósági pontszám (0-100)")
    plt.ylabel("Gyakoriság")
    plt.title("Hasonlósági pontszámok eloszlása")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    cumulative = np.cumsum(frequencies)
    plt.plot(bins, cumulative, marker="o", markersize=2, color="purple", linewidth=2)
    plt.axvline(
        x=percentile_90, color="red", linestyle="--", linewidth=2, label=f"90. percentilis: {percentile_90:.0f}"
    )
    plt.axvline(x=change_point, color="orange", linestyle="--", linewidth=2, label=f"Éles változás: {change_point}")
    plt.xlabel("Hasonlósági pontszám (0-100)")
    plt.ylabel("Kumulatív gyakoriság")
    plt.title("Kumulatív eloszlás")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    grad_values = [grad for _, grad in gradients]
    grad_positions = [i for i, _ in gradients]
    plt.plot(grad_positions, grad_values, color="darkred", linewidth=2, marker="o", markersize=3)
    plt.axvline(
        x=change_point, color="orange", linestyle="--", linewidth=2, label=f"Legnagyobb változás: {change_point}"
    )
    plt.xlabel("Hasonlósági pontszám (0-100)")
    plt.ylabel("Változás mértéke")
    plt.title("Gyakoriság változásának mértéke")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    top_10_threshold = int(percentile_90)
    top_bins = [i for i in bins if i >= top_10_threshold]
    top_frequencies = [counter.get(i, 0) for i in top_bins]

    plt.bar(top_bins, top_frequencies, alpha=0.8, color="lightcoral", edgecolor="darkred", width=0.8)
    plt.axvline(x=change_point, color="orange", linestyle="--", linewidth=2, label=f"Éles változás: {change_point}")
    plt.xlabel("Hasonlósági pontszám (0-100)")
    plt.ylabel("Gyakoriság")
    plt.title(f"Felső 10% részletes nézet (>={top_10_threshold})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"A grafikon elmentve: {output_filename}")

    print("\n=== ÖSSZESÍTETT STATISZTIKÁK ===")
    print(f"Összesen dokumentumok száma: {len(scores_100)}")
    print(f"Átlag pontszám: {mean_score:.2f}")
    print(f"Medián pontszám: {np.median(scores_array):.2f}")
    print(f"Szórás: {std_score:.2f}")
    print(f"Min pontszám: {min(scores_100)}")
    print(f"Max pontszám: {max(scores_100)}")
    print(f"90. percentilis: {percentile_90:.2f}")
    print(f"95. percentilis: {np.percentile(scores_array, 95):.2f}")
    print(f"99. percentilis: {np.percentile(scores_array, 99):.2f}")
    print(f"Legélesebb változás pontja: {change_point}")
    print(f"Felső 10%-ba tartozó dokumentumok száma: {sum(1 for s in scores_100 if s >= percentile_90)}")

    return {
        "scores": scores_100,
        "percentile_90": percentile_90,
        "change_point": change_point,
        "mean": mean_score,
        "std": std_score,
        "total_docs": len(scores_100),
    }


if __name__ == "__main__":
    query = "Mit csináljak ha elhagytam a diákigazolványom?"

    try:
        results = analyze_and_plot_similarity_scores(query, "similarity_analysis.png")
        print("\nElemzés sikeresen befejezve!")

    except Exception as e:
        print(f"Hiba történt: {e}")
        import traceback

        traceback.print_exc()
