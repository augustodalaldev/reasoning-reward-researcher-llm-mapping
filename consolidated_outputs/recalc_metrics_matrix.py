import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    experiment: str,
    plots_dir: Path,
) -> None:
    """Gera e salva heatmap da matriz de confusão em valores absolutos."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Mapeamento para exibição no gráfico
    label_map = {"ALTA": "HIGH", "MEDIA": "MEDIUM", "BAIXA": "LOW"}
    display_labels = [label_map.get(lbl, lbl) for lbl in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(len(labels) * 2 + 2, len(labels) * 2))
    
    sns.heatmap(
        cm,
        annot=True, 
        fmt="d",          
        cmap="Blues",
        xticklabels=display_labels, 
        yticklabels=display_labels,  
        ax=ax,
        linewidths=0.5, 
        linecolor="grey",
        annot_kws={"size": 20}
    )

    ax.tick_params(axis='both', labelsize=14)

    
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.set_title(f"Confusion Matrix\n{experiment}", fontsize=15, pad=15)

    out = plots_dir / f"{experiment}_confusion_matrix.png"
    fig.savefig(out, bbox_inches='tight', dpi=300) 
    plt.close(fig)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    experiment: str = "experiment",
) -> dict:
    """Calcula as métricas de classificação."""
    valid_mask = [p in labels for p in y_pred]
    n_unknown  = len(valid_mask) - sum(valid_mask)

    y_true_f = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_f = [p for p, v in zip(y_pred, valid_mask) if v]

    if len(y_true_f) == 0:
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

    return metrics

def evaluate_experiment(
    predictions_parquet: Path,
    labels: list[str],
    mode: str,
    experiment: str,
    results_dir: Path,
    cm_dir: Path,
) -> dict:
    """Calcula métricas, salva JSON, CSV da matriz de confusão e plota o heatmap."""
    label_col = "nivel_classificacao"
    df = pd.read_parquet(predictions_parquet)

    if "predicted" not in df.columns:
        raise ValueError(f"Coluna 'predicted' não encontrada em {predictions_parquet}")
    if label_col not in df.columns:
        raise ValueError(f"Coluna '{label_col}' não encontrada em {predictions_parquet}")

    y_true = df[label_col].tolist()
    y_pred = df["predicted"].tolist()

    # 1. Calcula e salva métricas gerais (JSON)
    metrics = compute_metrics(
        y_true     = y_true,
        y_pred     = y_pred,
        labels     = labels,
        experiment = experiment,
    )

    out_path = results_dir / f"{mode}_{experiment}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    # 2. Processamento da Matriz de Confusão
    valid_mask = [p in labels for p in y_pred]
    y_true_f = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_f = [p for p, v in zip(y_pred, valid_mask) if v]
    
    total_samples = len(y_pred)
    valid_samples = sum(valid_mask)
    valid_rate = (valid_samples / total_samples) * 100 if total_samples > 0 else 0
    
    experiment_name = f"qwen_{mode}_{experiment}"

    # Pastas de saída
    csv_dir = cm_dir / "csvs"
    plots_dir = cm_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Gera o DataFrame e salva o CSV
    cm_array = confusion_matrix(y_true_f, y_pred_f, labels=labels)
    cm_df = pd.DataFrame(cm_array, index=labels, columns=labels)
    cm_df.index.name = 'Real'
    cm_df.columns.name = 'Predicted'
    
    output_csv = csv_dir / f"{experiment_name}_cm.csv"
    cm_df.to_csv(output_csv)
    
    # Logs no terminal
    print(f"\n" + "="*60)
    print(f"MODELO: {experiment_name}")
    print(f"Respostas com formato válido: {valid_samples} de {total_samples} ({valid_rate:.2f}%) ({total_samples - valid_samples} não consideradas)")
    
    # Plota a imagem da matriz
    plot_confusion_matrix(
        y_true=y_true_f, 
        y_pred=y_pred_f, 
        labels=labels, 
        experiment=experiment_name, 
        plots_dir=plots_dir
    )
    
    print("="*60)
    print(cm_df.to_markdown())
    print("\n")

    return metrics


# --- Loop de Execução Principal ---
pasta_alvo = Path("results_default_labeled")
cm_base_dir = Path("confusion_matrix_default_labeled")

arquivos_parquet = glob.glob(os.path.join(pasta_alvo, "*.parquet"))

if not arquivos_parquet:
    print(f"Nenhum arquivo .parquet encontrado na pasta '{pasta_alvo}'.")
else:
    for caminho in arquivos_parquet:
        caminho_path = Path(caminho)
        nome_arquivo = caminho_path.name
        
        # Parse do nome para descobrir mode, experiment e labels
        core = nome_arquivo.replace("qwen_", "").replace("_test_predictions.parquet", "")
        
        if "2class" in core:
            mode = "2class"
            experiment = core.replace("_2class", "")
            labels = ["BAIXA", "ALTA"]
        elif "3class" in core:
            mode = "3class"
            experiment = core.replace("_3class", "")
            labels = ["BAIXA", "MEDIA", "ALTA"]
        else:
            print(f"Aviso: O arquivo '{nome_arquivo}' não segue o padrão esperado e foi pulado.")
            continue
            
        # Executa tudo para o arquivo atual
        evaluate_experiment(
            predictions_parquet = caminho_path,
            labels              = labels,
            mode                = mode,
            experiment          = experiment,
            results_dir         = pasta_alvo,
            cm_dir              = cm_base_dir
        )

    print(f"Processamento finalizado. Verifique as pastas '{pasta_alvo}' e '{cm_base_dir}'.")