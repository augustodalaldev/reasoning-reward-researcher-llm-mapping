# Label Correction and Metrics Recalculation Pipeline

This directory contains the scripts and results of the post-processing applied to the predictions made by the models (Base, SFT, GRPO, etc.).

Due to the maximum token limit set during response generation, some inferences failed to properly extract the `answer` tag. This resulted in empty values (`""`) in the `predicted` column of the result files. The pipeline documented here applies a fallback for these cases and recalculates the evaluation.

## What was done

1. **Default Label Assignment:** Empty values in the `predicted` column of the raw `.parquet` files were replaced with the `"BAIXA" (low)` class. No other rows were altered.
2. **Metrics Calculation:** New classification reports (Accuracy, F1-Macro, F1-Weighted, Precision, Recall) were generated, automatically detecting whether the experiment belonged to the 2-class (`"low"`, `"high"`) or 3-class (`"low"`, `"medium"`, `"high"`) approach.
3. **Confusion Matrices:** Updated confusion matrices were generated in two formats: tabular data for analysis and heatmaps for visualization.

## Folder Structure

* **`results_old/`**
  Contains the original `.parquet` files with the raw model inferences (including empty predictions).

* **`results_default_labeled/`**
  * Corrected `.parquet` files, where `predicted == ""` was replaced with `"low" ("BAIXA")`.
  * `.json` files with the recalculated evaluation metrics for each experiment (e.g., `3class_grpo_distance_metrics.json`).

* **`confusion_matrix_default_labeled/`**
  * **`csvs/`**: Confusion matrices saved in `.csv` tabular format (e.g., `qwen_2class_grpo_standard_cm.csv`).
  * **`plots/`**: Confusion matrix heatmaps generated with Seaborn and saved as `.png`.
