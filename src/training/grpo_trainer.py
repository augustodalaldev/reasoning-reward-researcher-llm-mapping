"""
src/training/grpo_trainer.py
=============================
GRPO (Group Relative Policy Optimization) usando Unsloth + TRL.

Segue estritamente a lógica do notebook oficial Unsloth para modelos Qwen3.5:
  - Usa GRPOConfig + GRPOTrainer da TRL.
  - Reward functions verificam:
      (a) formato correto da resposta (apenas um token de rótulo válido).
      (b) acurácia — rótulo previsto == rótulo de referência.
  - Captura mean_reward por passo para gerar curvas no paper.

Referência: https://github.com/unslothai/unsloth/blob/main/unsloth/notebooks/
"""

import json
import logging
import re
from pathlib import Path
from typing import Callable
import datasets

import pandas as pd


from src.utils.config import get_adapter_name, get_labels, get_output_dir, load_datasets, load_prompts, get_model_prefix
from src.utils.logger import setup_logger
from src.utils.resource_monitor import ResourceMonitor

logger = logging.getLogger("grpo_trainer")

# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _extract_xml_tag(text: str, tag: str) -> str:
    """Extrai o conteúdo entre tags XML. Retorna string vazia se falhar."""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag in text and end_tag in text:
        return text.split(start_tag)[-1].split(end_tag)[0].strip()
    return ""

def _make_format_reward(valid_labels: set[str]) -> Callable:
    """
    Recompensa de formato (+1.0): O modelo deve usar as tags <think> e <answer>,
    e o conteúdo de <answer> deve ser estritamente um dos rótulos válidos.
    """
    def format_reward(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for text in completions:
            if "<think>" in text and "</think>" in text and "<answer>" in text and "</answer>" in text:
                answer = _extract_xml_tag(text, "answer").upper()
                rewards.append(1.0 if answer in valid_labels else 0.0)
            else:
                rewards.append(0.0)
        return rewards
    return format_reward


def _make_accuracy_reward(cfg: dict, valid_labels: set[str]) -> Callable:
    """
    Retorna a função de recompensa de acurácia com base no config.yaml.
    Suporta modo 'standard' (0 ou 2) e 'distance' (0, 1, ou 2 baseado na proximidade) (deve ser configurada no config.yaml).
    """
    acc_mode = cfg.get("grpo", {}).get("rwfn_accuracy_mode", "standard")
    class_mode = cfg["classification"]["mode"]

    # Mapa ordinal para calcular distância
    ordinal_map = {"BAIXA": 0, "ALTA": 2}
    # Caso seja 3 class
    if class_mode == "3class":
        #ordinal_map = {"BAIXA": 0, "MEDIA": 1, "ALTA": 2}
        ordinal_map["MEDIA"] = 1

    def accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
        rewards = []
        for text, ref_label in zip(completions, answer):
            pred_ans = _extract_xml_tag(text, "answer").upper()
            ref_ans = ref_label.upper()

            # Se nem produziu um label válido, 0 pontos
            if pred_ans not in valid_labels:
                rewards.append(0.0)
                continue

            # Lógica de Distância (Apenas faz sentido no modo 3class)
            if acc_mode == "distance":
                dist = abs(ordinal_map[pred_ans] - ordinal_map[ref_ans])
                if dist == 0:
                    rewards.append(2.0)   # Acerto exato
                elif dist == 1:
                    rewards.append(0.5)   # Erro adjacente (ex: previu MEDIA, era ALTA) (3 classes apenas)
                else:
                    rewards.append(-1.0)  # Erro absurdo (ex: previu ALTA, era BAIXA)
            
            # Lógica Padrão (Tudo ou Nada)
            else:
                rewards.append(2.0 if pred_ans == ref_ans else 0.0)
                
        return rewards
    return accuracy_reward

def _make_reasoning_reward() -> Callable:
    """
    Recompensa de raciocínio (0.0 a 0.5): Calcula o overlap de palavras-chave 
    entre o bloco <think> gerado e a 'justificativa_final' do dataset professor.
    Induz o modelo a aprender QUAIS elementos justificam a classe.
    """
    def reasoning_reward(completions: list[str], gt_justificativa: list[str], **kwargs) -> list[float]:
        rewards = []
        for text, gt_just in zip(completions, gt_justificativa):
            think_block = _extract_xml_tag(text, "think").lower()
            
            # Extrai palavras com 4+ letras para evitar stopwords (o, a, um, com, etc)
            gt_words = set(re.findall(r'\w{4,}', str(gt_just).lower()))
            if not gt_words:
                rewards.append(0.0)
                continue
                
            think_words = set(re.findall(r'\w{4,}', think_block))
            overlap = len(gt_words.intersection(think_words))
            
            # Bônus proporcional ao acerto de palavras-chave da justificativa real
            score = overlap / len(gt_words) * 0.5
            rewards.append(min(0.5, score))
        return rewards
    return reasoning_reward


# ---------------------------------------------------------------------------
# Formatação de prompts (estilo Qwen3.5 chat)
# ---------------------------------------------------------------------------


def _build_grpo_dataset(df: pd.DataFrame, cfg: dict, tokenizer) -> "datasets.Dataset":
    """
    Cria o dataset GRPO estruturado com <think> e <answer>.
    Mantém o 'gt_justificativa' acessível para as reward functions.
    """
    
    ds = datasets.Dataset.from_pandas(df)
    
    # prompts carregados dinamicamente
    prompts = load_prompts(cfg)
    system_msg = prompts["system_prompt"]
    user_template = prompts["user_prompt_template"]
    
    def apply_grpo_format(example):
        user_content = user_template.format(
            nome_tema=example["nome_tema"],
            palavras_chave_tema=example["palavras_chave_tema"],
            nome_producao_perfil=example["nome_producao_perfil"],
            resumo_perfil=example["resumo_perfil"],
            palavras_chave_perfil=example["palavras_chave_perfil"],
            area_concentracao_perfil=example["area_concentracao_perfil"]
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        
        # Deixa o Unsloth pronto para injetar a resposta (add_generation_prompt=True)
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": prompt_str,
            "answer": example["nivel_classificacao"],
            "gt_justificativa": example.get("justificativa_final", ""),
        }

    ds = ds.map(apply_grpo_format, batched=False, desc="Formatando GRPO")
    
    # O GRPOTrainer repassa colunas sobressalentes como kwargs para as Reward Functions!
    cols_to_keep = ["prompt", "answer", "gt_justificativa"]
    cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
    return ds.remove_columns(cols_to_remove)



# ---------------------------------------------------------------------------
# Callback para capturar mean_reward
# ---------------------------------------------------------------------------

class RewardLoggerCallback:
    """Callback simples que coleta mean_reward do log_history do GRPOTrainer."""

    def __init__(self):
        self.reward_history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            self.reward_history.append({
                "step":        state.global_step,
                "mean_reward": logs["reward"],
            })
        return control


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def run_grpo(cfg: dict, resume_from_checkpoint: bool = False) -> dict:
    """
    Executa o treinamento GRPO.

    Returns
    -------
    dict com: mean_reward_history, resource_stats, adapter_path.
    """
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    logs_dir    = Path(cfg["paths"]["logs_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    setup_logger("grpo_trainer", logs_dir, level=cfg["logging"]["level"])

    mode         = cfg["classification"]["mode"]
    output_dir   = get_output_dir(cfg, "grpo")
    adapter_name = get_adapter_name(cfg, "grpo")
    valid_labels = set(get_labels(cfg))

    logger.info("=" * 60)
    logger.info("Iniciando GRPO | mode=%s | adapter=%s", mode, adapter_name)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Carrega splits (usa apenas treino para GRPO)
    # ------------------------------------------------------------------

    train_df, val_df, _ = load_datasets(cfg)
    logger.info("Treino: %d | Val: %d (GRPO usa apenas treino)", len(train_df), len(val_df))

    # ------------------------------------------------------------------
    # 2. Modelo + LoRA via Unsloth
    # ------------------------------------------------------------------
    model_cfg  = cfg["model"]
    lora_cfg   = cfg["lora"]
    grpo_cfg   = cfg["grpo"]

    with ResourceMonitor("grpo_training", n_samples=len(train_df)) as mon:

        logger.info("Carregando sft_cot mergeado: %s", model_cfg["grpo_name"])
        # ignora o processor (tokenizer) fornecido pelo unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = model_cfg["grpo_name"],
            max_seq_length = model_cfg["max_seq_length"],
            dtype          = None, #propio unsloth decide
            load_in_4bit   = model_cfg["load_in_4bit"],
            #Parametros vLLM
            fast_inference = True,
            gpu_memory_utilization = 0.60
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r                          = lora_cfg["r"],
            lora_alpha                 = lora_cfg["lora_alpha"],
            lora_dropout               = lora_cfg["lora_dropout"],
            target_modules             = lora_cfg["target_modules"],
            bias                       = lora_cfg["bias"],
            use_gradient_checkpointing = lora_cfg["use_gradient_checkpointing"],
            random_state               = lora_cfg["random_state"],
            use_rslora                 = lora_cfg["use_rslora"],
        )
        logger.info("Modelo + LoRA prontos para GRPO.")

        # ------------------------------------------------------------------
        # 3. Dataset GRPO
        # ------------------------------------------------------------------
        train_dataset = _build_grpo_dataset(train_df, cfg, tokenizer)
        logger.info("Dataset GRPO formatado: %d amostras", len(train_dataset))

        # ------------------------------------------------------------------
        # 4. Reward functions
        # ------------------------------------------------------------------
        format_reward_fn   = _make_format_reward(valid_labels)
        accuracy_reward_fn = _make_accuracy_reward(cfg, valid_labels)
        reasoning_reward_fn = _make_reasoning_reward()

        # ------------------------------------------------------------------
        # 5. GRPOConfig — alinhado ao notebook Unsloth para Qwen3.5
        # ------------------------------------------------------------------
        grpo_config = GRPOConfig(
            output_dir                  = str(output_dir),
            num_train_epochs            = grpo_cfg["num_train_epochs"],
            per_device_train_batch_size = grpo_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps = grpo_cfg["gradient_accumulation_steps"],
            learning_rate               = grpo_cfg["learning_rate"],
            max_grad_norm               = grpo_cfg["max_grad_norm"],
            warmup_ratio                = grpo_cfg["warmup_ratio"],
            lr_scheduler_type           = grpo_cfg["lr_scheduler_type"],
            bf16                        = grpo_cfg["bf16"] and is_bfloat16_supported(),
            fp16                        = grpo_cfg["fp16"] and not is_bfloat16_supported(),
            optim                       = grpo_cfg["optim"],
            report_to                   = grpo_cfg["report_to"],
            # GRPO-específico
            num_generations             = grpo_cfg["num_generations"],
            max_prompt_length           = grpo_cfg["max_prompt_length"],
            max_completion_length       = grpo_cfg["max_completion_length"],
            temperature                 = grpo_cfg["temperature"],
            beta                        = grpo_cfg["beta"],
            logging_steps               = grpo_cfg["logging_steps"],
            save_steps                  = grpo_cfg["save_steps"],
            save_total_limit            = grpo_cfg["save_total_limit"],
            seed                        = cfg["project"]["seed"],
        )

        # ------------------------------------------------------------------
        # 6. Callback de logging de reward
        # ------------------------------------------------------------------
        reward_callback = RewardLoggerCallback()

        # Adaptação para TrainerCallback da HuggingFace
        class _HFRewardCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                reward_callback.on_log(args, state, control, logs=logs, **kwargs)
                return control

        # ------------------------------------------------------------------
        # 7. GRPOTrainer
        # ------------------------------------------------------------------
        
        # funções adaptadas tanto para modo 3class e 2class
        reward_fns = [format_reward_fn, accuracy_reward_fn, reasoning_reward_fn]
            
        trainer = GRPOTrainer(
            model          = model,
            processing_class = tokenizer,
            args           = grpo_config,
            train_dataset  = train_dataset,
            reward_funcs   = reward_fns,
            callbacks      = [_HFRewardCallback()],
        )

        # ------------------------------------------------------------------
        # 8. Treino com monitoramento de recursos
        # ------------------------------------------------------------------
        
        logger.info("Iniciando treinamento GRPO…")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    resource_stats = mon.stats
    resource_stats.log_summary(logger)

    # ------------------------------------------------------------------
    # 9. Salva métricas
    # ------------------------------------------------------------------
    mean_reward_history = reward_callback.reward_history

    # Também extrai do log_history do trainer (mais robusto)
    for entry in trainer.state.log_history:
        if "reward" in entry:
            exists = any(
                r["step"] == entry.get("step") for r in mean_reward_history
            )
            if not exists:
                mean_reward_history.append({
                    "step":        entry.get("step", 0),
                    "mean_reward": entry["reward"],
                })

    mean_reward_history.sort(key=lambda x: x["step"])

    # nomeando corretamente
    model_prefix = get_model_prefix(cfg)

    acc_mode = cfg["grpo"].get("rwfn_accuracy_mode", "standard")
    exp_name = f"{model_prefix}_grpo_{acc_mode}_{mode}"


    # salva tanto mean_reward_history 
    metrics = {
        "experiment":          exp_name,
        "adapter":             adapter_name,
        "mean_reward_history": mean_reward_history,
        "resource_stats":      resource_stats.to_dict(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{exp_name}_training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    logger.info("Métricas GRPO salvas: %s", metrics_path)

    # ------------------------------------------------------------------
    # 10. Salva adapter
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Adapter GRPO salvo em: %s", output_dir)

    # ------------------------------------------------------------------
    # 10. Salva o modelo Merged para vLLM
    # ------------------------------------------------------------------
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # logger.info("Adapter GRPO salvo em: %s", output_dir)
    # model.save_pretrained(str(output_dir))
    # tokenizer.save_pretrained(str(output_dir))
    # logger.info("Fundindo LoRA no modelo base (16-bit) para vLLM...")
    # try:
    #     # Funde para 16-bit (ideal para máxima compatibilidade no vLLM)
    #     model.save_pretrained_merged(
    #         str(output_dir), tokenizer, save_method="merged_16bit"
    #     )
    #     logger.info("Modelo fundido salvo com sucesso para vLLM em: %s", output_dir)
    # except Exception as e:
    #     logger.error("Erro ao fundir o modelo: %s. Salvando apenas o adapter foi salvo.", e)
        


    return metrics