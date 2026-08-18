"""
src/inference/engine.py
========================
Motor de inferência acelerado com vLLM.

Suporta:
  - base        : prompt direto zero-shot.
  - few_shot    : prompt com N exemplos por classe.
  - sft         : carrega LoRA adapter salvo localmente.
  - sft_cot     : carrega LoRA adapter salvo localmente.
  - grpo        : carrega LoRA adapter de RL.

Características:
  - Usa apply_chat_template nativo do tokenizer para evitar erros de sintaxe.
  - Extração robusta do rótulo lendo tags XML <answer>.
  - Resultados salvos em .parquet com predições e ground-truth.
"""

import gc
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional
import time

import pandas as pd

from src.utils.config import get_labels, get_output_dir, load_datasets, load_prompts, get_model_prefix
from src.utils.logger import setup_logger
from src.utils.resource_monitor import ResourceMonitor

logger = logging.getLogger("engine")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
APPROACH_MODES = {"base", "few_shot", "sft", "sft_cot", "grpo"}


# ---------------------------------------------------------------------------
# Extração de rótulo da resposta do modelo
# ---------------------------------------------------------------------------

# Dicionário de mapeamento para flexões e erros comuns (já sem acentos)
LABEL_MAPPING = {
    # --- Variações de ALTA ---
    "ALTO": "ALTA",
    "ALTAS": "ALTA",
    "ALTOS": "ALTA",

    # --- Variações de MEDIA ---
    "MEDIO": "MEDIA",
    "MEDIAS": "MEDIA",
    "MEDIOS": "MEDIA",
    "MEDIANA": "MEDIA",
    "MEDIANO": "MEDIA",
    "MEDIANAS": "MEDIA",
    "MEDIANOS": "MEDIA",

    # --- Variações de BAIXA ---
    "BAIXO": "BAIXA",
    "BAIXAS": "BAIXA",
    "BAIXOS": "BAIXA",
    "BAIXISSIMA": "BAIXA", 
    "BAIXISSIMO": "BAIXA",

    # --- Variações de NAO-ALTA (Modo 2 classes) ---
    "NAO ALTA": "NAO-ALTA",
    "NAOALTA": "NAO-ALTA",
    "NAO-ALTAS": "NAO-ALTA",
    "NAO ALTAS": "NAO-ALTA",
    "NAOALTAS": "NAO-ALTA",
    "NAO-ALTO": "NAO-ALTA",
    "NAO ALTO": "NAO-ALTA",
    "NAOALTO": "NAO-ALTA",
    "NAO-ALTOS": "NAO-ALTA",
    "NAO ALTOS": "NAO-ALTA",
    "NAOALTOS": "NAO-ALTA"
}

def _resolve_label(found_label: str) -> str:
    """Mapeia a label encontrada de volta para a label oficial."""
    return LABEL_MAPPING.get(found_label, found_label)

def _remove_accents(text: str) -> str:
    """Remove acentos da string para facilitar o match (ex: MÉDIA -> MEDIA)."""
    return "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def _filter_extract(text: str, valid_labels: list[str]) -> tuple[str, list[str]]:
    """
    Filtragem comum a ambos os métodos de extração

    Returns:
        (text, search_targets)
        tuple[str, list[str]]
    """

    # Garante que as labels oficiais estão limpas, sem acento e em maiúsculo
    valid_labels_upper = [_remove_accents(l.upper()) for l in valid_labels]
    
    # Expande os alvos de busca: labels oficiais + chaves do mapeamento
    # Ex: ["ALTA", "MEDIA", "BAIXA", "MEDIO", "ALTO", "BAIXO", ...]
    search_targets = valid_labels_upper + [k for k, v in LABEL_MAPPING.items() if v in valid_labels_upper]

    def resolve_label(found_label: str) -> str:
        """Mapeia a label encontrada de volta para a label oficial."""
        return LABEL_MAPPING.get(found_label, found_label)

    # Remove acentos do texto inteiro logo de cara para evitar dor de cabeça com regex
    text = _remove_accents(text)

    return (text, search_targets)

def _extract_label_reasoning(text: str, valid_labels: list[str]) -> str:
    """
    Tenta extrair da tag <answer> primeiro, com sanitização e mapeamento de sinônimos.
    Se falhar, faz busca isolada dentro da tag, e por fim busca no texto inteiro.
    """
    text, search_targets = _filter_extract(text, valid_labels)

    # 1. Busca a tag <answer> com regex (captura quebras de linha)
    match_tag = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    
    if match_tag:
        ans_raw = match_tag.group(1).strip().upper()
        
        # Sanitização: Remove o que NÃO for letra, número, hífen ou espaço
        ans_clean = re.sub(r'[^A-Z0-9- ]', '', ans_raw).strip()
        
        # 1.1 Match exato do conteúdo limpo da tag (agora aceita sinônimos mapeados)
        if ans_clean in search_targets:
            return _resolve_label(ans_clean)
            
        # 1.2 Regex APENAS dentro do que estava na tag
        for label in search_targets:
            pattern = re.compile(r"\b" + re.escape(label) + r"\b", re.IGNORECASE)
            if pattern.search(ans_raw):
                return _resolve_label(label)
                
        logger.debug("Tag <answer> encontrada, mas rótulo inválido após limpeza: %r", ans_raw)

    # --- Fallbacks (Sem a tag <answer>) ---
    text_upper = text.strip().upper()
    
    # 2. Resposta limpa: match exato no texto inteiro 
    text_clean = re.sub(r'[^A-Z0-9- ]', '', text_upper).strip()
    if text_clean in search_targets:
        return _resolve_label(text_clean)

    # 3. Busca o primeiro token válido usando regex no texto inteiro
    for label in search_targets:
        pattern = re.compile(r"\b" + re.escape(label) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            return _resolve_label(label)


    logger.debug("Rótulo não encontrado no texto completo: %r", text[:80])
    return ""

def _extract_label_base(text: str, valid_labels: list[str]) -> str:
    """
    Extração da label predita para métodos que não utilizam reasoning
    """
    text, search_targets = _filter_extract(text, valid_labels)

    text_upper = text.strip().upper()
    
    # 2. Resposta limpa: match exato no texto inteiro 
    text_clean = re.sub(r'[^A-Z0-9- ]', '', text_upper).strip()
    if text_clean in search_targets:
        return _resolve_label(text_clean)

    # 3. Busca o primeiro token válido usando regex no texto inteiro
    for label in search_targets:
        pattern = re.compile(r"\b" + re.escape(label) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            return _resolve_label(label)
        
    logger.debug("Rótulo não encontrado: %r", text[:80])
    return ""

# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Encapsula a lógica de inferência com vLLM para todas as abordagens.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._llm = None
        self._is_adapter = False

    # ------------------------------------------------------------------
    # Inicialização do motor vLLM (lazy)
    # ------------------------------------------------------------------

    def _init_vllm(self, model_path: str, is_adapter: bool = False) -> None:
        """
        Inicializa o LLM vLLM.
        """
        from vllm import LLM
        import torch

        infer_cfg = self.cfg["inference"]

        logger.info("Inicializando vLLM com model_path=%s", model_path)

        kwargs = dict(
            model                  = model_path,
            tensor_parallel_size   = infer_cfg["tensor_parallel_size"],
            gpu_memory_utilization = infer_cfg["gpu_memory_utilization"],
            max_model_len          = infer_cfg["max_model_len"],
            trust_remote_code      = True,
            dtype                  = torch.bfloat16,
            enforce_eager          = True,   # obrigatório para BnB
            quantization           = "bitsandbytes",
            load_format            = "bitsandbytes",
        )

        if is_adapter:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = self.cfg["lora"]["r"]

        self._llm = LLM(**kwargs)
        self._is_adapter = is_adapter
        logger.info("vLLM inicializado com sucesso.")

    def _cleanup_vllm(self):
        """Destrói a instância do vLLM e limpa a VRAM."""
        if self._llm is not None:
            from vllm.distributed.parallel_state import destroy_model_parallel
            
            logger.info("Descarregando vLLM e limpando VRAM...")
            destroy_model_parallel()
            del self._llm
            self._llm = None
            
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            
            # Um pequeno delay garante que o SO sincronize a liberação de memória
            time.sleep(2) 
            logger.info("VRAM liberada.")

    def _get_sampling_params(self):
        from vllm import SamplingParams
        infer_cfg = self.cfg["inference"]
        experiment = self.cfg["classification"]["experiment"]

        if experiment in ["base", "few_shot", "sft"]:
            max_tokens = infer_cfg["max_new_tokens"]
            stop_tokens = ["<|im_end|>"]
        else:  # grpo, sft_cot
            max_tokens = self.cfg.get("grpo", {}).get("max_completion_length", infer_cfg["max_new_tokens"])
            stop_tokens = ["</answer>", "<|im_end|>"]

        if experiment in ["grpo", "sft_cot"]:
            rep_penalty = 1.0 # Desativa para não quebrar a lógica
        else:
            rep_penalty = infer_cfg["repetition_penalty"]

        return SamplingParams(
            temperature        = infer_cfg["temperature"],
            top_p              = infer_cfg["top_p"],
            max_tokens         = max_tokens,
            repetition_penalty = rep_penalty,
            stop               = stop_tokens,   
        )

    # ------------------------------------------------------------------
    # Construção de prompts usando Tokenizer Nativo
    # ------------------------------------------------------------------

    def _build_prompts(self, df: pd.DataFrame) -> list[str]:
        prompts_cfg = load_prompts(self.cfg)
        system_msg = prompts_cfg["system_prompt"]
        user_template = prompts_cfg["user_prompt_template"]
        experiment = self.cfg["classification"]["experiment"]

        tokenizer = self._llm.get_tokenizer()
        prompts = []

        for _, row in df.iterrows():
            user_content = user_template.format(
                nome_tema=row["nome_tema"],
                palavras_chave_tema=row["palavras_chave_tema"],
                nome_producao_perfil=row["nome_producao_perfil"],
                resumo_perfil=row["resumo_perfil"],
                palavras_chave_perfil=row["palavras_chave_perfil"],
                area_concentracao_perfil=row["area_concentracao_perfil"]
            )
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ]
            
            # ativação de thinking dinamico
            enable_thinking = experiment in ["grpo", "sft_cot"]

            chat_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            prompts.append(chat_prompt)

        return prompts

    # ------------------------------------------------------------------
    # Inferência em lotes
    # ------------------------------------------------------------------

    def _batch_generate(self, prompts: list[str], adapter_path: Optional[str] = None) -> list[str]:
        from vllm.lora.request import LoRARequest

        sampling_params = self._get_sampling_params()
        
        lora_request = None
        if adapter_path and self._is_adapter:
            lora_request = LoRARequest("classifier_adapter", 1, adapter_path)

        logger.info("Enviando %d prompts para o motor vLLM (Continuous Batching ativado)...", len(prompts))
        
        outputs = self._llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=True
        )
        
        return [out.outputs[0].text if out.outputs else "" for out in outputs]

    # ------------------------------------------------------------------
    # Função pública principal
    # ------------------------------------------------------------------

    def run(
        self,
        approach: str,
        adapter_path: Optional[str] = None,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Executa inferência completa para a abordagem especificada.
        """
        if approach not in APPROACH_MODES:
            raise ValueError(f"Abordagem inválida: {approach!r}. Use: {APPROACH_MODES}")

        # Injeta a abordagem temporariamente
        self.cfg["classification"]["experiment"] = approach
        
        # Pega os labels validos AGORA, apos injetar a abordagem e modo
        valid_labels = get_labels(self.cfg)

        logs_dir    = Path(self.cfg["paths"]["logs_dir"])
        results_dir = Path(self.cfg["paths"]["results_dir"])
        mode        = self.cfg["classification"]["mode"]

        setup_logger("inference_engine", logs_dir, level=self.cfg["logging"]["level"])
        logger.info("=" * 60)
        logger.info("Inferência | approach=%s | mode=%s", approach, mode)
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # 1. Carrega splits
        # ------------------------------------------------------------------
        _, _, test_df = load_datasets(self.cfg)
        eval_df = test_df.copy()
        logger.info("Teste: %d amostras", len(eval_df))

        # ------------------------------------------------------------------
        # 2. Lógica de caminhos de modelo (Adapters vs Merged)
        # ------------------------------------------------------------------
        is_adapter = approach in ["sft", "sft_cot", "grpo"]


        try:
            # ------------------------------------------------------------------
            # 3. Gera predições com monitoramento
            # ------------------------------------------------------------------
            with ResourceMonitor(f"inference_{approach}", n_samples=len(eval_df)) as mon:

                if is_adapter:
                    base_model = self.cfg["model"]["name"]
                    adapter_path = adapter_path or str(get_output_dir(self.cfg, "sft" if approach in ["sft", "sft_cot"] else "grpo"))
                    if not Path(adapter_path).exists():
                        raise FileNotFoundError(f"Adapter não encontrado: {adapter_path}")
                    
                    self._init_vllm(base_model, is_adapter=True)
                    
                else:
                    base_model = self.cfg["model"]["name"]
                    adapter_path = None
                    self._init_vllm(base_model, is_adapter=False)


                prompts = self._build_prompts(eval_df)

                # realiza inferencia
                raw_responses = self._batch_generate(prompts, adapter_path=adapter_path)

            resource_stats = mon.stats
            resource_stats.log_summary(logger)

            # ------------------------------------------------------------------
            # 4. Extrai rótulos
            # ------------------------------------------------------------------
            is_reasoning = approach in ["sft_cot", "grpo"]
            if is_reasoning: 
                predictions = [_extract_label_reasoning(r, valid_labels) for r in raw_responses]
            else: 
                predictions = [_extract_label_base(r, valid_labels) for r in raw_responses]

            eval_df["raw_response"] = raw_responses
            eval_df["predicted"]    = predictions

            # ------------------------------------------------------------------
            # 5. Salva resultados
            # ------------------------------------------------------------------
            if save_results:
                results_dir.mkdir(parents=True, exist_ok=True)
                
                suffix = ""
                if approach == "grpo":
                    rw_fn_mode = self.cfg.get("grpo", {}).get("rwfn_accuracy_mode", "standard")
                    suffix = f"_{rw_fn_mode}"
                    
                model_prefix = get_model_prefix(self.cfg)
                file_prefix = f"{model_prefix}_{approach}{suffix}_{mode}"
                
                output_path = results_dir / f"{file_prefix}_test_predictions.parquet"
                eval_df.to_parquet(output_path, index=False)
                logger.info("Predições salvas: %s", output_path)

                stats_path = results_dir / f"{file_prefix}_resource_stats.json"
                resource_stats.save_json(stats_path)

        finally:
            # Garante que a memória da GPU seja limpa mesmo se o vLLM der um erro
            self._cleanup_vllm()

        return eval_df