"""
Reporting and visualization utilities.

Generates publication-quality figures for probing results,
similarity matrices, shift analyses, and property emergence maps.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union


def generate_report(
    results: Any,
    save: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Auto-dispatch to the right visualization based on result type.
    """
    # Import here to avoid matplotlib dependency at import time
    from matscope.core import ProbeResult, SimilarityResult, ShiftResult

    if isinstance(results, ProbeResult):
        _plot_probe_result(results, save=save, show=show, figsize=figsize)
    elif isinstance(results, dict) and all(isinstance(v, ProbeResult) for v in results.values()):
        _plot_emergence_map(results, save=save, show=show, figsize=figsize)
    elif isinstance(results, SimilarityResult):
        _plot_similarity(results, save=save, show=show, figsize=figsize)
    elif isinstance(results, ShiftResult):
        _plot_shift(results, save=save, show=show, figsize=figsize)
    else:
        raise TypeError(f"Cannot generate report for type: {type(results)}")


def _plot_probe_result(result, save=None, show=True, figsize=(10, 6)):
    """Plot probing accuracy across layers."""
    import matplotlib.pyplot as plt
    import numpy as np

    df = result.to_dataframe()
    metric = "accuracy" if "accuracy" in df.columns else "r2"
    std_col = f"{metric}_std" if f"{metric}_std" in df.columns else None

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))

    bars = ax.bar(x, df[metric], color="#2196F3", alpha=0.8, edgecolor="white")
    if std_col:
        ax.errorbar(x, df[metric], yerr=df[std_col], fmt="none", color="black", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["layer"], rotation=45, ha="right")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Probing '{result.target_property}' — {result.probe_type} probe")
    ax.set_ylim(0, 1.05 if metric == "accuracy" else None)

    # Highlight best layer
    best_idx = df[metric].idxmax()
    bars[best_idx].set_color("#FF5722")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_emergence_map(results_dict, save=None, show=True, figsize=(12, 6)):
    """Plot property emergence map — heatmap of accuracy across layers x properties."""
    import matplotlib.pyplot as plt
    import numpy as np

    properties = list(results_dict.keys())
    layers = list(next(iter(results_dict.values())).layer_results.keys())
    metric = "accuracy" if "accuracy" in next(iter(next(iter(results_dict.values())).layer_results.values())) else "r2"

    matrix = np.zeros((len(properties), len(layers)))
    for i, prop in enumerate(properties):
        for j, layer in enumerate(layers):
            if layer in results_dict[prop].layer_results:
                matrix[i, j] = results_dict[prop].layer_results[layer].get(metric, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1 if metric == "accuracy" else None)

    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(properties)))
    ax.set_yticklabels(properties)

    # Annotate cells
    for i in range(len(properties)):
        for j in range(len(layers)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > 0.7 else "black")

    ax.set_title("Property Emergence Map")
    ax.set_xlabel("Layer (depth →)")
    ax.set_ylabel("Physical Property")
    plt.colorbar(im, label=metric.capitalize())

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_similarity(result, save=None, show=True, figsize=(8, 8)):
    """Plot CKA/CCA similarity matrix."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(result.similarity_matrix, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(result.layer_names_b)))
    ax.set_xticklabels(result.layer_names_b, rotation=45, ha="right")
    ax.set_yticks(range(len(result.layer_names_a)))
    ax.set_yticklabels(result.layer_names_a)
    ax.set_title(f"Representation Similarity ({result.method.upper()})")
    ax.set_xlabel("Model B layers")
    ax.set_ylabel("Model A layers")
    plt.colorbar(im, label="Similarity")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_shift(result, save=None, show=True, figsize=(10, 5)):
    """Plot distribution shift across layers."""
    import matplotlib.pyplot as plt
    import numpy as np

    layers = result.layers_analyzed
    scores = [result.shift_scores[l] for l in layers]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layers))
    colors = ["#FF5722" if l == result.most_affected_layer else "#2196F3" for l in layers]

    ax.bar(x, scores, color=colors, alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel(f"Shift Magnitude ({result.metadata.get('method', 'unknown')})")
    ax.set_title("Distribution Shift by Layer")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
