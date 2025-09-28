import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append(os.path.join(os.getcwd(), "server"))
from core.config.config import config
from dal.mongo_db import MongoRetriever

DEFAULT_SCORE_BAND_HINT = (0.78, 0.88)


def compute_generality_hint(std_score, theoretical_std=None, observed_range=None, score_band_hint=None):
    """Ad hoc normalizált általánossági mutató 0 és 1 között."""

    noise_floor = 0.015  # apró varianciák kiszűrése
    target_std = 0.18  # ennél magasabb szórás már nagyon általános promptot jelez

    std_norm = np.clip((std_score - noise_floor) / max(target_std, 1e-6), 0.0, 1.0)

    if theoretical_std is not None:
        theory_norm = np.clip((theoretical_std - noise_floor) / max(target_std, 1e-6), 0.0, 1.0)
    else:
        theory_norm = std_norm

    range_norm = std_norm
    if observed_range is not None:
        min_effective_range = 0.01
        expected_range = max(score_band_hint[1] - score_band_hint[0], min_effective_range) if score_band_hint else 0.25
        range_norm = np.clip((observed_range - min_effective_range) / expected_range, 0.0, 1.0)

    combined = 0.5 * std_norm + 0.3 * theory_norm + 0.2 * range_norm
    return float(np.clip(combined, 0.0, 1.0))


def fit_beta_distribution(scores_01):
    """Beta-eloszlás illesztése a score-okra MLE módszerrel."""
    scores_modified = np.clip(scores_01, 1e-6, 1 - 1e-6)
    alpha_est, beta_est, _, _ = stats.beta.fit(scores_modified, floc=0, fscale=1)
    return alpha_est, beta_est


def calculate_optimal_topk(
    scores,
    beta_threshold=0.90,
    adaptive_threshold=True,
    generality_hint=None,
    score_band_hint=None,
):
    """Intelligensebb top-k kiválasztás variancia alapú korlátokkal."""

    sorted_scores = np.sort(scores)[::-1]
    sample_std = float(np.std(scores))
    observed_range = float(sorted_scores[0] - sorted_scores[-1]) if len(sorted_scores) else 0.0
    generality_score = (
        float(np.clip(generality_hint, 0.0, 1.0))
        if generality_hint is not None
        else compute_generality_hint(sample_std, observed_range=observed_range, score_band_hint=score_band_hint)
    )

    def _dynamic_bounds(population_size: int) -> dict[str, int | float]:
        if population_size == 0:
            return {"min": 0, "max": 0}

        min_fraction = float(np.interp(generality_score, [0.0, 1.0], [0.002, 0.10]))
        max_fraction = float(np.interp(generality_score, [0.0, 1.0], [0.035, 0.28]))

        min_k = max(3, int(np.ceil(population_size * min_fraction)))
        adaptive_min_cap = int(np.ceil(6 + generality_score * 28))
        min_k = int(max(3, min(min_k, adaptive_min_cap)))

        max_by_fraction = max(min_k, int(np.ceil(population_size * max_fraction)))
        max_by_scale = max(min_k, int(np.ceil(np.sqrt(population_size) * (0.8 + 0.9 * generality_score))))
        max_k = int(min(population_size, max(max_by_fraction, max_by_scale)))

        return {"min": min_k, "max": max_k}

    if adaptive_threshold:
        # 1. Adaptív küszöb: percentilis alapú
        high_percentile = np.percentile(scores, 95)
        medium_percentile = np.percentile(scores, 75)

        # 2. Gradiens alapú elválasztás (legnagyobb csökkenés)
        score_diffs = np.diff(sorted_scores)
        gradient_change_idx = np.argmin(score_diffs) + 1 if len(score_diffs) else 1

        # 3. Standard deviation alapú küszöb
        mean_score = float(np.mean(scores))
        std_threshold = mean_score + sample_std
        std_based_k = int(np.sum(sorted_scores >= std_threshold))

        # 4. Percentilis alapú
        percentile_based_k = int(np.sum(scores >= high_percentile))
        percentile_75_k = int(np.sum(scores >= medium_percentile))

        # 5. Hagyományos kumulatív tömeg
        cumsum_scores = np.cumsum(sorted_scores)
        total_mass = float(np.sum(sorted_scores))
        threshold_mass = beta_threshold * total_mass
        cumulative_k = int(np.argmax(cumsum_scores >= threshold_mass) + 1)

        raw_candidates = {
            "gradient": gradient_change_idx,
            "std_based": std_based_k,
            "percentile_95": percentile_based_k,
            "percentile_75": percentile_75_k,
            "cumulative": cumulative_k,
        }

        bounds = _dynamic_bounds(len(scores))

        clipped_candidates = {}
        adjusted_values = []
        for name, k_val in raw_candidates.items():
            if k_val <= 0:
                continue
            clipped_val = int(np.clip(k_val, bounds["min"], bounds["max"]))
            clipped_candidates[name] = clipped_val
            adjusted_values.append(clipped_val)

        if not adjusted_values:
            adjusted_values.append(bounds["min"])

        if generality_score <= 0.35:
            optimal_k = int(min(adjusted_values))
        elif generality_score >= 0.75:
            optimal_k = int(max(adjusted_values))
        else:
            optimal_k = int(np.median(adjusted_values))

        metadata = {
            "raw_candidates": raw_candidates,
            "clipped_candidates": clipped_candidates,
            "bounds": bounds,
            "generality_score": generality_score,
            "sample_std": sample_std,
            "observed_range": observed_range,
        }

        return optimal_k, cumsum_scores, total_mass, metadata

    # Eredeti kumulatív tömeg módszer korlátokkal
    cumsum_scores = np.cumsum(sorted_scores)
    total_mass = float(np.sum(sorted_scores))
    threshold_mass = beta_threshold * total_mass
    top_k = int(np.argmax(cumsum_scores >= threshold_mass) + 1)

    bounds = _dynamic_bounds(len(scores))
    bounded_top_k = int(np.clip(top_k, bounds["min"], bounds["max"]))

    metadata = {
        "raw_candidates": {"cumulative": top_k},
        "clipped_candidates": {"cumulative": bounded_top_k},
        "bounds": bounds,
        "generality_score": generality_score,
        "sample_std": sample_std,
        "observed_range": observed_range,
    }

    return bounded_top_k, cumsum_scores, total_mass, metadata


def bootstrap_topk_confidence(
    scores,
    n_bootstrap=1000,
    beta_threshold=0.90,
    confidence=0.95,
    score_band_hint=None,
):
    """Bootstrap módszerrel konfidencia-intervallum számítása a top-k-ra."""
    bootstrap_topks = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
        top_k, _, _, _ = calculate_optimal_topk(
            bootstrap_sample,
            beta_threshold,
            adaptive_threshold=True,
            score_band_hint=score_band_hint,
        )
        bootstrap_topks.append(top_k)

    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    ci_lower = np.percentile(bootstrap_topks, lower_percentile)
    ci_upper = np.percentile(bootstrap_topks, upper_percentile)

    return np.array(bootstrap_topks), ci_lower, ci_upper


def analyze_and_plot_similarity_scores(
    query: str,
    output_filename: str = "beta_analysis.png",
    score_band_hint: tuple[float, float] | None = DEFAULT_SCORE_BAND_HINT,
):
    """Elemzi és plotálja a hasonlósági pontszámokat Beta-eloszlás illesztéssel."""
    print(f"Elemzés indítása: '{query}'")

    retriever = MongoRetriever(config.mongo.COLLECTION_NAME)
    print("Hasonlósági pontszámok lekérése...")
    results = retriever.get_all_similarity_scores(query)

    if not results:
        print("Nincsenek eredmények!")
        return

    all_scores = np.array([result["similarity_score"] for result in results])

    # Intelligensebb szűrés
    # 1. Dinamikus küszöb: medián - 1*std vagy 0.01, amelyik nagyobb
    dynamic_threshold = max(0.01, np.median(all_scores) - np.std(all_scores))
    scores_01 = all_scores[all_scores >= dynamic_threshold]

    print(f"Dinamikus küszöb: {dynamic_threshold:.3f}")
    print(
        f"Szűrt dokumentumok száma: {len(scores_01)} / {len(all_scores)} ({len(scores_01) / len(all_scores) * 100:.1f}%)"
    )

    if len(scores_01) == 0:
        print("Nincsenek megfelelő score-ok!")
        return

    observed_range = float(scores_01.max() - scores_01.min())
    print(f"Score tartomány: {scores_01.min():.3f} - {scores_01.max():.3f}")
    print(f"Score szórás: {np.std(scores_01):.4f} (eredeti: {np.std(all_scores):.4f})")  # Alapstatisztikák
    mean_score = np.mean(scores_01)
    variance_score = np.var(scores_01)
    std_score = np.sqrt(variance_score)

    print("\n=== ALAPSTATISZTIKÁK ===")
    print(f"Átlag (μ): {mean_score:.4f}")
    print(f"Variancia (σ²): {variance_score:.4f}")
    print(f"Szórás (σ): {std_score:.4f}")

    # Beta-eloszlás illesztése
    print("\n=== BETA-ELOSZLÁS ===")
    theoretical_var = None
    theoretical_std = None

    try:
        alpha, beta = fit_beta_distribution(scores_01)
        print(f"α paraméter: {alpha:.4f}")
        print(f"β paraméter: {beta:.4f}")

        theoretical_mean = alpha / (alpha + beta)
        theoretical_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        theoretical_std = float(np.sqrt(theoretical_var))
        print(f"Elméleti átlag: {theoretical_mean:.4f}")
        print(f"Elméleti variancia: {theoretical_var:.4f}")

        ks_stat, ks_pvalue = stats.kstest(scores_01, lambda x: stats.beta.cdf(x, alpha, beta))
        print(f"KS teszt p-érték: {ks_pvalue:.4f}")

    except Exception as e:
        print(f"Beta illesztés hiba: {e}")
        alpha, beta = None, None
        theoretical_std = None
        ks_pvalue = None

    generality_hint = compute_generality_hint(
        std_score,
        theoretical_std,
        observed_range=observed_range,
        score_band_hint=score_band_hint,
    )
    print(f"Generáltsági pontszám: {generality_hint:.2f} (0 → specifikus, 1 → általános)")

    # Optimális top-k
    print("\n=== TOP-K SZÁMÍTÁS ===")
    optimal_topk = None
    cumsum_scores = None
    total_mass = None
    topk_metadata = None

    for beta_thresh in [0.80, 0.85, 0.90, 0.95]:
        (
            top_k,
            current_cumsum,
            current_total_mass,
            metadata,
        ) = calculate_optimal_topk(
            scores_01,
            beta_thresh,
            generality_hint=generality_hint,
            score_band_hint=score_band_hint,
        )

        coverage_percent = (current_cumsum[top_k - 1] / current_total_mass) * 100
        print(f"β = {beta_thresh}: top-k = {top_k}, lefedettség = {coverage_percent:.1f}%")

        if beta_thresh == 0.90:
            optimal_topk = top_k
            cumsum_scores = current_cumsum
            total_mass = current_total_mass
            topk_metadata = metadata

    if optimal_topk is None:
        optimal_topk, cumsum_scores, total_mass, topk_metadata = calculate_optimal_topk(
            scores_01,
            0.90,
            generality_hint=generality_hint,
            score_band_hint=score_band_hint,
        )

    if topk_metadata:
        bounds = topk_metadata.get("bounds", {})
        clipped = topk_metadata.get("clipped_candidates", {})
        print(
            "Dinamikus korlátok: min={min_k}, max={max_k} (generalitás={gen:.2f})".format(
                min_k=bounds.get("min"),
                max_k=bounds.get("max"),
                gen=topk_metadata.get("generality_score", 0.0),
            )
        )
        if clipped:
            candidate_str = ", ".join(f"{name}: {value}" for name, value in clipped.items())
            print(f"Korrigált jelöltek: {candidate_str}")

    # Bootstrap
    print("\n=== BOOTSTRAP ===")
    bootstrap_topks, ci_lower, ci_upper = bootstrap_topk_confidence(
        scores_01,
        1000,
        0.90,
        score_band_hint=score_band_hint,
    )
    print(f"Optimális top-k: {optimal_topk}")
    print(f"95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]")
    print(f"Bootstrap átlag: {np.mean(bootstrap_topks):.1f} ± {np.std(bootstrap_topks):.1f}")

    coverage_at_optimal = (cumsum_scores[optimal_topk - 1] / total_mass) * 100
    bounds = topk_metadata.get("bounds", {"min": None, "max": None}) if topk_metadata else {"min": None, "max": None}
    prompt_type_label = "Magas variancia → Általános" if generality_hint >= 0.55 else "Alacsony variancia → Specifikus"

    # Plotolás
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Eredeti vs illesztett eloszlás
    ax1.hist(scores_01, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="navy", label="Eredeti adatok")

    if alpha is not None and beta is not None:
        x_range = np.linspace(0.001, 0.999, 1000)
        fitted_pdf = stats.beta.pdf(x_range, alpha, beta)
        ax1.plot(x_range, fitted_pdf, "r-", linewidth=3, label=f"Beta({alpha:.2f}, {beta:.2f})")

    ax1.axvline(mean_score, color="green", linestyle="--", linewidth=2, label=f"Átlag: {mean_score:.3f}")
    ax1.set_xlabel("Hasonlósági pontszám (0-1)")
    ax1.set_ylabel("Sűrűség")
    ax1.set_title("Eredeti vs Illesztett Beta-eloszlás")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Kumulatív tömeg
    sorted_indices = np.arange(1, len(scores_01) + 1)
    cumulative_mass_percent = (cumsum_scores / total_mass) * 100

    ax2.plot(sorted_indices, cumulative_mass_percent, "b-", linewidth=2, label="Kumulatív tömeg")
    ax2.axhline(y=90, color="red", linestyle="--", linewidth=2, label="90% küszöb")
    ax2.axvline(x=optimal_topk, color="orange", linestyle="--", linewidth=2, label=f"Optimális top-k: {optimal_topk}")
    ax2.set_xlabel("Top-k")
    ax2.set_ylabel("Kumulatív tömeg (%)")
    ax2.set_title("Kumulatív tömeg és optimális top-k")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Bootstrap
    ax3.hist(bootstrap_topks, bins=30, alpha=0.7, color="lightcoral", edgecolor="darkred", density=True)
    ax3.axvline(optimal_topk, color="blue", linestyle="-", linewidth=3, label=f"Eredeti top-k: {optimal_topk}")
    ax3.axvline(ci_lower, color="red", linestyle="--", linewidth=2, label=f"95% CI: [{ci_lower:.0f}, {ci_upper:.0f}]")
    ax3.axvline(ci_upper, color="red", linestyle="--", linewidth=2)
    ax3.set_xlabel("Top-k érték")
    ax3.set_ylabel("Sűrűség")
    ax3.set_title("Bootstrap konfidencia-intervallum")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statisztikák
    ax4.axis("off")
    stats_text = f"""STATISZTIKÁK

Adatok: {len(scores_01)}
Átlag: {mean_score:.4f}
Variancia: {variance_score:.4f}
Szórás: {std_score:.4f}
Generáltsági pontszám: {generality_hint:.2f}
Dinamikus top-k tartomány: [{bounds.get("min")}, {bounds.get("max")}]

Beta paraméterek:"""

    if alpha is not None and beta is not None:
        stats_text += f"""
α = {alpha:.3f}
β = {beta:.3f}
KS p-érték: {ks_pvalue:.4f}"""
        if theoretical_std is not None:
            stats_text += f"\nElméleti szórás: {theoretical_std:.4f}"
    else:
        stats_text += "\nIllesztés sikertelen"

    stats_text += f"""

Optimális top-k: {optimal_topk}
CI: [{ci_lower:.0f}, {ci_upper:.0f}]
Lefedettség: {coverage_at_optimal:.1f}%
Dinamikus korlátok: [{bounds.get("min")}, {bounds.get("max")}]
Prompt típus: {prompt_type_label}"""

    ax4.text(
        0.05,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nGrafikon mentve: {output_filename}")

    return {
        "scores_01": scores_01,
        "mean": mean_score,
        "variance": variance_score,
        "std": std_score,
        "alpha": alpha,
        "beta": beta,
        "optimal_topk": optimal_topk,
        "topk_ci": (ci_lower, ci_upper),
        "total_docs": len(scores_01),
        "ks_pvalue": ks_pvalue,
        "generality_score": generality_hint,
        "topk_bounds": bounds,
        "topk_metadata": topk_metadata,
        "coverage_at_optimal": coverage_at_optimal,
        "clipped_candidates": topk_metadata.get("clipped_candidates") if topk_metadata else None,
        "observed_range": observed_range,
    }


if __name__ == "__main__":
    query = "Students requirements"

    try:
        results = analyze_and_plot_similarity_scores(query, "beta_distribution_analysis.png")
        print("\n✅ Elemzés kész!")

        if results:
            print(f"🎯 Optimális top-k: {results['optimal_topk']}")
            generality = results.get("generality_score")
            if generality is not None:
                prompt_type = "általános" if generality >= 0.55 else "specifikus"
                print(f"📊 Prompt jellege: {prompt_type} (generáltság = {generality:.2f})")
            bounds = results.get("topk_bounds") or {}
            if bounds:
                print(f"📈 Dinamikus top-k tartomány: [{bounds.get('min')}, {bounds.get('max')}]")

    except Exception as e:
        print(f"❌ Hiba: {e}")
        import traceback

        traceback.print_exc()
