"""
src/training/sft_trainer.py
============================
Supervised Fine-Tuning (SFT) usando Unsloth + LoRA sobre Qwen3.5-9B-bnb-4bit.

Pipeline:
  1. Carrega modelo base via Unsloth (4-bit).
  2. Aplica adaptadores LoRA.
  3. Formata os splits de treino/val em datasets HuggingFace.
  4. Executa SFTTrainer (TRL).
  5. Salva o adapter e as métricas de loss + recursos.
"""

# tentativa de regular memória
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
import time
from pathlib import Path
from typing import Optional
import datasets

import pandas as pd


from src.utils.config import get_adapter_name, get_output_dir, load_datasets, load_prompts, get_model_prefix
from src.utils.logger import setup_logger
from src.utils.resource_monitor import ResourceMonitor

logger = logging.getLogger("sft_trainer")

# ---------------------------------------------------------------------------
# Formatação de exemplos para o SFTTrainer
# ---------------------------------------------------------------------------


def _format_dataset(df: pd.DataFrame, cfg: dict, tokenizer) -> datasets.Dataset:
    """
    Converte um DataFrame em HuggingFace Dataset aplicando o chat_template do Qwen3.5.
    """

    # 1. Converte direto para Dataset do HF
    dataset = datasets.Dataset.from_pandas(df)
    
    # 2. Resgata os prompts do config.yaml
    # carregamento dinamico dos prompts
    prompts = load_prompts(cfg)
    system_msg = prompts["system_prompt"]
    user_template = prompts["user_prompt_template"]
    
    experiment = cfg["classification"]["experiment"]
    enable_thinking = experiment == 'sft_cot'
    
    def apply_chat_format(example):
        # Preenche o template do usuário com as múltiplas colunas
        # O ft_user no config.yaml deve ter os placeholders correspondentes (ex: {nome_tema})
        user_content = user_template.format(
            nome_tema=example["nome_tema"],
            palavras_chave_tema=example["palavras_chave_tema"],
            nome_producao_perfil=example["nome_producao_perfil"],
            resumo_perfil=example["resumo_perfil"],
            palavras_chave_perfil=example["palavras_chave_perfil"],
            area_concentracao_perfil=example["area_concentracao_perfil"]
        )
        
        # Constrói o nome da chave dinamicamente: "sft_assistant" ou "sft_cot_assistant"
        assistant_key = f"{experiment}_assistant"
        
        # Caso seja sft_cot, passa a justificativa E a classe
        if experiment == 'sft_cot':
            assistant_content = cfg["prompts"][assistant_key].format(
                justificativa_final=example.get("justificativa_final", ""),
                nivel_classificacao=example["nivel_classificacao"]
            )
        # SFT Padrão, passa apenas a classe
        else:
            assistant_content = cfg["prompts"][assistant_key].format(
                nivel_classificacao=example["nivel_classificacao"]
            )
            
        messages = [
                {"role": "system",    "content": system_msg},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
        ]

        # Aplica o chat_template aqui — gera string pronta com EOS
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking
        )

        return {"text": text}
        
    dataset = dataset.map(
        apply_chat_format,
        batched=False,
        remove_columns=dataset.column_names,  # remove tudo, fica só messages
    )

    return dataset


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def run_sft(cfg: dict,  resume_from_checkpoint: bool = False) -> dict:
    """
    Executa o treinamento SFT.

    Parameters
    ----------
    cfg : configuração completa.
    resume_from_checkpoint : continua de um checkpoint salvo.

    Returns
    -------
    dict com métricas: train_loss_history, eval_loss_history, resource_stats.
    """
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    logs_dir   = Path(cfg["paths"]["logs_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    setup_logger("sft_trainer", logs_dir, level=cfg["logging"]["level"])

    mode = cfg["classification"]["mode"]
    output_dir  = get_output_dir(cfg, 'sft')
    adapter_name = get_adapter_name(cfg, 'sft')
    experiment = cfg["classification"]["experiment"]

    logger.info("=" * 60)
    logger.info("Iniciando SFT | mode=%s | adapter=%s", mode, adapter_name)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Carrega splits
    # ------------------------------------------------------------------
    train_df, val_df, _ = load_datasets(cfg)
    logger.info("Treino: %d | Val: %d", len(train_df), len(val_df))

    # ------------------------------------------------------------------
    # 2. Carrega modelo + LoRA via Unsloth
    # ------------------------------------------------------------------
    model_cfg  = cfg["model"]
    lora_cfg   = cfg["lora"]
    sft_cfg    = cfg["sft"]

    logger.info("Carregando modelo base: %s", model_cfg["name"])
    # ignora o processor (tokenizer) fornecido pelo unsloth

    n_train = len(train_df)
    with ResourceMonitor("sft_training", n_samples=n_train) as mon:

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = model_cfg["name"],
            max_seq_length = model_cfg["max_seq_length"],
            dtype          = model_cfg["dtype"],
            load_in_4bit   = model_cfg["load_in_4bit"],
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r                        = lora_cfg["r"],
            lora_alpha               = lora_cfg["lora_alpha"],
            lora_dropout             = lora_cfg["lora_dropout"],
            target_modules           = lora_cfg["target_modules"],
            bias                     = lora_cfg["bias"],
            use_gradient_checkpointing = lora_cfg["use_gradient_checkpointing"],
            random_state             = lora_cfg["random_state"],
            use_rslora               = lora_cfg["use_rslora"],
        )
        logger.info("Modelo + LoRA prontos.")

        # ------------------------------------------------------------------
        # 3. Formata datasets
        # ------------------------------------------------------------------

        train_dataset = _format_dataset(train_df, cfg, tokenizer)
        val_dataset   = _format_dataset(val_df,   cfg, tokenizer)
        logger.info("Datasets formatados: train=%d | val=%d", len(train_dataset), len(val_dataset))

        # ------------------------------------------------------------------
        # 4. SFTConfig
        # ------------------------------------------------------------------

        training_args = SFTConfig(
            output_dir                  = str(output_dir),
            max_seq_length              = model_cfg["max_seq_length"],
            num_train_epochs            = sft_cfg["num_train_epochs"],
            per_device_train_batch_size = sft_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size  = sft_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps = sft_cfg["gradient_accumulation_steps"],
            learning_rate               = sft_cfg["learning_rate"],
            warmup_ratio                = sft_cfg["warmup_ratio"],
            lr_scheduler_type           = sft_cfg["lr_scheduler_type"],
            weight_decay                = sft_cfg["weight_decay"],
            fp16                        = sft_cfg["fp16"],
            bf16                        = sft_cfg["bf16"],
            logging_steps               = sft_cfg["logging_steps"],
            eval_strategy               = "steps",
            eval_steps                  = sft_cfg["eval_steps"],
            save_steps                  = sft_cfg["save_steps"],
            save_total_limit            = sft_cfg["save_total_limit"],
            load_best_model_at_end      = sft_cfg["load_best_model_at_end"],
            metric_for_best_model       = sft_cfg["metric_for_best_model"],
            report_to                   = sft_cfg["report_to"],
            optim                       = sft_cfg["optim"],
            seed                        = cfg["project"]["seed"],
            packing                     = sft_cfg["packing"],
        )

        # ------------------------------------------------------------------
        # 5. SFTTrainer
        # ------------------------------------------------------------------
        trainer = SFTTrainer(
            model           = model,
            processing_class= tokenizer,
            train_dataset   = train_dataset,
            eval_dataset    = val_dataset,
            args            = training_args,
            dataset_text_field = "text"
        )

        # ------------------------------------------------------------------
        # 6. Treino com monitoramento de recursos
        # ------------------------------------------------------------------

        logger.info("Iniciando treinamento SFT…")
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint or None
        )

    resource_stats = mon.stats
    resource_stats.log_summary(logger)

    # ------------------------------------------------------------------
    # 7. Extrai histórico de loss
    # ------------------------------------------------------------------
    log_history = trainer.state.log_history
    train_losses = [
        {"step": e["step"], "loss": e["loss"]}
        for e in log_history if "loss" in e and "eval_loss" not in e
    ]
    eval_losses = [
        {"step": e["step"], "eval_loss": e["eval_loss"]}
        for e in log_history if "eval_loss" in e
    ]

    # ------------------------------------------------------------------
    # 8. Salva métricas
    # ------------------------------------------------------------------
    model_prefix = get_model_prefix(cfg)
    exp_name = f"{model_prefix}_{experiment}_{mode}"

    metrics = {
        "experiment": exp_name,
        "adapter": adapter_name,
        "train_loss_history": train_losses,
        "eval_loss_history":  eval_losses,
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "resource_stats": resource_stats.to_dict(),
    }

    metrics_path = results_dir / f"{exp_name}_training_metrics.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    logger.info("Métricas de treino salvas: %s", metrics_path)

    # ------------------------------------------------------------------
    # 9. Salva adapter
    # ------------------------------------------------------------------

    if experiment == "sft_cot":
        logger.info("Experimento SFT-CoT: Salvando adapter LoRA")
        # SEMPRE salva o adapter primeiro — nunca perde o treino
        adapter_backup_dir = output_dir / "lora_adapters"
        model.save_pretrained(str(adapter_backup_dir))
        tokenizer.save_pretrained(str(adapter_backup_dir))
        logger.info("Adapter LoRA salvo em: %s", adapter_backup_dir)

        # Não é necessário
        # logger.info("Tentando fundir LoRA no modelo base (merged_4bit)...")
        # try:
        #     model.save_pretrained_merged(
        #         str(output_dir), tokenizer, save_method="merged_4bit_forced"
        #     )
        #     logger.info("Merge concluído com sucesso.")
        # except RuntimeError as e:
        #     logger.warning("Merge merged_4bit_forced falhou (%s). Tentando lora_merged_16bit...", e)
        #     try:
        #         model.save_pretrained_merged(
        #             str(output_dir), tokenizer, save_method="lora_merged_16bit"
        #         )
        #         logger.info("Merge lora_merged_16bit concluído.")
        #     except Exception as e2:
        #         logger.error(
        #             "Todos os métodos de merge falharam (%s). "
        #             "Use o adapter salvo em: %s",
        #             e2, adapter_backup_dir
        #         )
    else:
        logger.info("Experimento SFT Padrão: Salvando apenas o adaptador LoRA...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

   

    return metrics