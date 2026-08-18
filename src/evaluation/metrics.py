"""
src/evaluation/metrics.py
==========================
Cálculo de métricas científicas e geração de gráficos para o paper.

Métricas calculadas:
  - Accuracy, F1-macro, F1-weighted, Precision-macro, Recall-macro
  - Relatório de classificação completo (por classe)
  - Matriz de confusão (absoluta e normalizada)

Gráficos gerados:
  - Matriz de confusão (heatmap)
  - Curvas de loss SFT (train vs eval)
  - Curva de mean_reward GRPO
  - Barras de comparação de F1-macro entre abordagens
"""


import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Backend sem display para servidores headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cálculo de métricas
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    experiment: str = "experiment",
) -> dict:
    """
    Calcula todas as métricas de classificação.

    Parameters
    ----------
    y_true      : rótulos verdadeiros.
    y_pred      : rótulos preditos.
    labels      : lista ordenada de rótulos válidos.
    experiment  : nome do experimento (para logging).

    Returns
    -------
    dict com todas as métricas escalares e o relatório por classe.
    """
    # Filtra amostras com predição ""
    valid_mask = [p in labels for p in y_pred]
    n_unknown  = len(valid_mask) - sum(valid_mask)
    if n_unknown > 0:
        logger.warning(
            "[%s] %d amostras com predição "" (%.1f%%) serão excluídas das métricas.",
            experiment, n_unknown, 100 * n_unknown / len(y_pred)
        )

    y_true_f = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_f = [p for p, v in zip(y_pred, valid_mask) if v]

    if len(y_true_f) == 0:
        logger.error("Nenhuma predição válida para calcular métricas!")
        return {}

    metrics = {
        "experiment":         experiment,
        "n_total":            len(y_true),
        "n_unknown":          n_unknown,
        "n_evaluated":        len(y_true_f),
        "accuracy":           round(accuracy_score(y_true_f, y_pred_f), 4),
        "f1_macro":           round(f1_score(y_true_f, y_pred_f, labels=labels, average="macro",    zero_division=0), 4),
        "f1_weighted":        round(f1_score(y_true_f, y_pred_f, labels=labels, average="weighted", zero_division=0), 4),
        "precision_macro":    round(precision_score(y_true_f, y_pred_f, labels=labels, average="macro",    zero_division=0), 4),
        "recall_macro":       round(recall_score(y_true_f, y_pred_f, labels=labels, average="macro",    zero_division=0), 4),
        "classification_report": classification_report(
            y_true_f, y_pred_f, labels=labels, output_dict=True, zero_division=0
        ),
    }

    logger.info(
        "[%s] Acc=%.4f | F1-macro=%.4f | F1-weighted=%.4f",
        experiment, metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"]
    )

    return metrics


def evaluate_experiment(
    predictions_parquet: str | Path,
    cfg: dict,
    experiment: Optional[str] = None,
    save_metrics: bool = True,
) -> dict:
    """
    Carrega predições salvas, calcula métricas e salva JSON de resultados.
    """
    from src.utils.config import get_labels

    results_dir = Path(cfg["paths"]["results_dir"])
    label_col   = "nivel_classificacao"
    labels      = get_labels(cfg)
    experiment = experiment or cfg["classification"]["experiment"]
    mode        = cfg["classification"]["mode"]

    df = pd.read_parquet(predictions_parquet)

    if "predicted" not in df.columns:
        raise ValueError(f"Coluna 'predicted' não encontrada em {predictions_parquet}")
    if label_col not in df.columns:
        raise ValueError(f"Coluna '{label_col}' não encontrada em {predictions_parquet}")

    metrics = compute_metrics(
        y_true     = df[label_col].tolist(),
        y_pred     = df["predicted"].tolist(),
        labels     = labels,
        experiment = experiment,
    )

    if save_metrics:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{mode}_{experiment}_metrics.json"

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)

        logger.info("Métricas salvas: %s", out_path)

    return metrics


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _style():
    """Estilo global dos gráficos."""
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    experiment: str,
    plots_dir: str | Path,
    normalize: bool = False,
) -> None:
    """Gera e salva heatmap da matriz de confusão."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmin, vmax = ".2f", 0.0, 1.0
        title_suffix = "(normalizada)"
    else:
        cm_plot = cm
        fmt, vmin, vmax = "d", None, None
        title_suffix = "(absoluta)"

    fig, ax = plt.subplots(figsize=(len(labels) * 2 + 2, len(labels) * 2))
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        vmin=vmin, vmax=vmax, ax=ax,
        linewidths=0.5, linecolor="grey",
    )
    ax.set_xlabel("Predito",    fontsize=13)
    ax.set_ylabel("Verdadeiro", fontsize=13)
    ax.set_title(f"Matriz de Confusão {title_suffix}\n{experiment}", fontsize=14, pad=15)

    out = plots_dir / f"{experiment}_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Confusion matrix salva: %s", out)


def plot_loss_curves(
    metrics_json: str | Path,
    plots_dir: str | Path,
) -> None:
    """Plota curvas de loss (train/eval) de um experimento SFT."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_json, "r") as fh:
        data = json.load(fh)

    experiment = data.get("experiment", "sft")
    train_losses = data.get("train_loss_history", [])
    eval_losses  = data.get("eval_loss_history",  [])

    if not train_losses and not eval_losses:
        logger.warning("Sem dados de loss em %s", metrics_json)
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    if train_losses:
        if "epoch" in train_losses[0]:
            from itertools import groupby
            grouped = {}
            for e in train_losses:
                ep = round(e["epoch"])  # agrupa pelo epoch inteiro
                grouped.setdefault(ep, []).append(e["loss"])
            epochs = sorted(grouped)
            losses = [sum(grouped[ep]) / len(grouped[ep]) for ep in epochs]
            ax.plot(epochs, losses, label="Train Loss", color="#2196F3", lw=2)
            ax.set_xlabel("Epoch")

        else:
            steps  = [e["step"] for e in train_losses]
            losses = [e["loss"] for e in train_losses]
            ax.plot(steps, losses, label="Train Loss", color="#2196F3", lw=2)
            ax.set_xlabel("Step")

    if eval_losses:
        steps  = [e["step"] for e in eval_losses]
        losses = [e["eval_loss"] for e in eval_losses]
        ax.plot(steps, losses, label="Eval Loss", color="#F44336", lw=2, linestyle="--")

    
    ax.set_ylabel("Loss")
    ax.set_title(f"Curvas de Loss — {experiment}")
    ax.legend()

    out = plots_dir / f"{experiment}_loss_curves.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Loss curves salvas: %s", out)


def plot_grpo_reward(
    metrics_json: str | Path,
    plots_dir: str | Path,
) -> None:
    """Plota curva de mean_reward do GRPO."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_json, "r") as fh:
        data = json.load(fh)

    experiment   = data.get("experiment", "grpo")
    reward_hist  = data.get("mean_reward_history", [])

    if not reward_hist:
        logger.warning("Sem dados de mean_reward em %s", metrics_json)
        return

    steps   = [e["step"]        for e in reward_hist]
    rewards = [e["mean_reward"] for e in reward_hist]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, color="#4CAF50", lw=2)
    ax.fill_between(steps, rewards, alpha=0.15, color="#4CAF50")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Curva de Recompensa GRPO — {experiment}")

    out = plots_dir / f"{experiment}_mean_reward.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("GRPO reward curve salva: %s", out)


def plot_f1_comparison(
    metrics_list: list[dict],
    plots_dir: str | Path,
    title: str = "Comparação F1-macro entre Abordagens",
) -> None:
    """
    Gráfico de barras comparando F1-macro entre várias abordagens.

    Parameters
    ----------
    metrics_list : lista de dicts com chaves 'experiment' e 'f1_macro'.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_list:
        return

    experiments = [m["experiment"] for m in metrics_list]
    f1_scores   = [m.get("f1_macro", 0.0) for m in metrics_list]

    colors = sns.color_palette("colorblind", len(experiments))
    fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 1.8), 6))
    bars = ax.bar(experiments, f1_scores, color=colors, edgecolor="black", linewidth=0.7)

    # Anotações de valor
    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-macro")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")

    out = plots_dir / "f1_macro_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Gráfico de comparação F1 salvo: %s", out)


def plot_all(cfg: dict, all_metrics: list[dict]) -> None:
    """
    Gera todos os gráficos de uma vez a partir da lista de métricas de
    todos os experimentos.
    """
    _style()

    plots_dir   = Path(cfg["paths"]["plots_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    
    # CORREÇÃO 1: Coluna alvo hardcoded
    label_col   = "nivel_classificacao"
    mode        = cfg["classification"]["mode"]

    from src.utils.config import get_labels, get_model_prefix
    
    labels = get_labels(cfg)
    model_prefix = get_model_prefix(cfg)

    #  nomes de arquivos de Loss do SFT alinhados com o sft_trainer.py
    sft_metrics_path = results_dir / f"{model_prefix}_sft_{mode}_training_metrics.json"
    if sft_metrics_path.exists():
        plot_loss_curves(sft_metrics_path, plots_dir)
        
    sft_cot_metrics_path = results_dir / f"{model_prefix}_sft_cot_{mode}_training_metrics.json"
    if sft_cot_metrics_path.exists():
        plot_loss_curves(sft_cot_metrics_path, plots_dir)

    # CORREÇÃO 3: Busca a curva do GRPO dinamicamente independente do accuracy_mode
    grpo_files = list(results_dir.glob(f"{model_prefix}_grpo*_{mode}_training_metrics.json"))
    for grpo_metrics_path in grpo_files:
        plot_grpo_reward(grpo_metrics_path, plots_dir)

    # Matrizes de confusão por experimento
    for m in all_metrics:
        exp = m.get("experiment", "unknown")

        # Busca flexível em vez de path hardcoded
        matches = list(results_dir.glob(f"*{exp}*_test_predictions.parquet"))
        if not matches:
            logger.warning("Parquet de predições não encontrado para: %s", exp)
            continue
        pq_path = matches[0]

        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            if "predicted" in df.columns and label_col in df.columns:
                valid = df["predicted"].isin(labels)
                plot_confusion_matrix(
                    y_true     = df.loc[valid, label_col].tolist(),
                    y_pred     = df.loc[valid, "predicted"].tolist(),
                    labels     = labels,
                    experiment = exp,
                    plots_dir  = plots_dir,
                )

    # Comparação F1-macro
    eval_metrics = [m for m in all_metrics if "f1_macro" in m]
    if eval_metrics:
        plot_f1_comparison(eval_metrics, plots_dir)