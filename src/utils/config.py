"""
src/utils/config.py
Leitura, validação e resolução do config.yaml com suporte a overrides via CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

import yaml


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """
    Carrega o config.yaml.

    Parameters
    ----------
    config_path : caminho para o arquivo YAML.

    Returns
    -------
    dict com a configuração completa.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)

    # Garante que todos os diretórios-chave existam
    _ensure_directories(cfg)

    return cfg


def get_labels(cfg: dict) -> list[str]:
    """Retorna a lista de rótulos conforme o modo de classificação."""
    mode = cfg["classification"]["mode"]
    if mode == "3class":
        return cfg["classification"]["labels_3class"]
    elif mode == "2class":
        return cfg["classification"]["labels_2class"]
    else:
        raise ValueError(f"Modo de classificação inválido: {mode!r}. Use '3class' ou '2class'.")


def get_label_set(cfg: dict) -> set[str]:
    return set(get_labels(cfg))


def resolve_path(cfg: dict, *keys: str) -> Path:
    """
    Resolve um path dentro da config navegando pelas chaves fornecidas.

    Exemplo:
        resolve_path(cfg, "paths", "logs_dir")  →  Path("outputs/logs")
    """
    node = cfg
    for k in keys:
        node = node[k]
    return Path(node)

def get_model_prefix(cfg: dict) -> str:
    """Extrai um nome curto do modelo (ex: 'unsloth/Qwen3.5-9B' -> 'qwen')."""
    raw_name = cfg["model"]["name"].lower()
    if "qwen" in raw_name: return "qwen"
    if "llama" in raw_name: return "llama"
    if "gemma" in raw_name: return "gemma"
    if "mistral" in raw_name: return "mistral"
    # Fallback genérico: pega a última parte antes do traço
    return raw_name.split("/")[-1].split("-")[0]


def get_adapter_name(cfg: dict, section: str) -> str:
    """
    Substitui os placeholders {experiment} e {mode} no adapter_name pelo sufixo correto.
    Serve para qualquer seção (sft, grpo) e qualquer experimento.
    """
    mode = cfg["classification"]["mode"]
    experiment = cfg["classification"]["experiment"]
    model_prefix = get_model_prefix(cfg)
    
    raw_name = cfg[section]["adapter_name"]
    formatted_name = raw_name.replace("{experiment}", experiment).replace("{mode}", mode).replace("{model}", model_prefix)
    
    return formatted_name


def get_output_dir(cfg: dict, section: str) -> Path:
    """
    Substitui os placeholders {experiment} e {mode} no output_dir pelo sufixo correto.
    Serve para qualquer seção e experimento.
    """
    mode = cfg["classification"]["mode"]
    experiment = cfg["classification"]["experiment"]
    model_prefix = get_model_prefix(cfg)

    raw_dir = cfg[section]["output_dir"]
    formatted_dir = raw_dir.replace("{experiment}", experiment).replace("{mode}", mode).replace("{model}", model_prefix)
    
    return Path(formatted_dir)


# Função que carrega os datasets de acordo com o modo
def load_datasets(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os DataFrames de treino, validação e teste diretamente dos Parquets processados."""
    mode = cfg["classification"]["mode"] # '2class' ou '3class'
    
    train_path = Path(cfg["paths"][f"{mode}_training"])
    val_path   = Path(cfg["paths"][f"{mode}_validation"])
    test_path  = Path(cfg["paths"][f"{mode}_test"])
    
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Parquets não encontrados")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    return train_df, val_df, test_df


def load_prompts(cfg: dict) -> dict:
    """
    Monta dinamicamente os templates do System e User prompts 
    baseado no modo (2class/3class) e experimento.
    """
    mode = cfg["classification"]["mode"]
    experiment = cfg["classification"]["experiment"]

    # Validação de segurança
    valid_modes = ['2class', '3class']
    valid_experiments = ['base', 'few_shot', 'sft', 'sft_cot', 'grpo']
    
    if mode not in valid_modes or experiment not in valid_experiments:
        raise ValueError(f"Configuração inválida. Mode: {mode}, Exp: {experiment}")
    
    cfg_prompts = cfg["prompts"]

    # system prompt
    system_base = cfg_prompts["system_base"]
    system_raw = cfg_prompts[f"system_{mode}"]
    
    system_prompt = system_raw.format(system_base=system_base)

    # user prompt
    dados_template = cfg_prompts["dados_template"]

    # Escolhe a instrução correta baseado no experimento 
    if experiment in ['sft_cot', 'grpo']: #usam mesmo prompt
        instruction_raw = cfg_prompts["grpo_instruction"]
    else:
        instruction_raw = cfg_prompts["base_instruction"]

    # usa replace para injetar os dados sem acionar as chaves do dataset ({nome_tema}, etc)
    user_prompt_template = instruction_raw.replace("{dados}", dados_template)


    if experiment == 'few_shot':
        examples = cfg_prompts[f"few_shot_examples_{mode}"]
        user_prompt_template = cfg_prompts["few_shot_instruction"].format(
            examples=examples,
            instruction=user_prompt_template
        )

    return {
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template
    }


# ---------------------------------------------------------------------------
# Internos
# ---------------------------------------------------------------------------

def _ensure_directories(cfg: dict) -> None:
    """Cria os diretórios definidos em cfg['paths'] se não existirem."""
    paths_section: dict[str, Any] = cfg.get("paths", {})
    dir_keys = [k for k in paths_section if k.endswith("_dir")]
    for key in dir_keys:
        Path(paths_section[key]).mkdir(parents=True, exist_ok=True)

def _validate_label(label: str, possible_labels: list[str]):
    return label in possible_labels