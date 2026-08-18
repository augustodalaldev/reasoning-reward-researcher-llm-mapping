"""
main.py
========
Ponto de entrada (entrypoint) do projeto paper2026.

Orquestra a execução sequencial ou seletiva de:
  1. Inferência Baseline            (base)
  2. Inferência Few-shot            (few_shot)
  3. SFT Padrão (Treino)            (sft)
  4. SFT Padrão (Inferência)        (sft_infer)
  5. SFT-CoT (Treino)               (sft_cot)
  6. SFT-CoT (Inferência)           (sft_cot_infer)
  7. GRPO Standard (Treino)         (grpo_standard)
  8. GRPO Standard (Inferência)     (grpo_standard_infer)
  9. GRPO Distance (Treino)        (grpo_distance)
  10. GRPO Distance (Inferência)    (grpo_distance_infer)
  11. Avaliação e gráficos          (eval)

Uso rápido:
    python main.py --steps sft sft_infer eval
    python main.py --steps grpo_standard grpo_standard_infer eval
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
import copy

# ---------------------------------------------------------------------------
# Setup básico de logging antes de importar módulos pesados
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

from src.utils.config import load_config, get_model_prefix
from src.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ALL_STEPS = [
    "base",
    "few_shot",
    "sft",
    "sft_infer",
    "sft_cot",
    "sft_cot_infer",
    "grpo_standard",
    "grpo_standard_infer",
    "grpo_distance",
    "grpo_distance_infer",
    "eval",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_parquet_path(cfg: dict, approach: str) -> Path:
    """Constrói o caminho do parquet gerado pelo engine.py para a avaliação."""
    results_dir = Path(cfg["paths"]["results_dir"])
    mode = cfg["classification"]["mode"]
    model_prefix = get_model_prefix(cfg)
    
    suffix = ""
    if approach == "grpo":
        rw_fn_mode = cfg.get("grpo", {}).get("rwfn_accuracy_mode", "standard")
        suffix = f"_{rw_fn_mode}"
        
    return results_dir / f"{model_prefix}_{approach}{suffix}_{mode}_test_predictions.parquet"

def _run_inference_and_eval(cfg: dict, args: argparse.Namespace, approach: str) -> dict:
    """Função genérica para enxugar as chamadas de inferência e avaliação."""
    from src.inference.engine import InferenceEngine
    from src.evaluation.metrics import evaluate_experiment
    
    logger.info("[STEP] %s Inference", approach.upper())
    
    engine = InferenceEngine(cfg)
    engine.run(approach=approach, adapter_path=args.adapter_path)
    
    pq_path = _get_parquet_path(cfg, approach)
    
    eval_name = f"{approach}_{cfg.get('grpo', {}).get('rwfn_accuracy_mode', 'standard')}" if approach == "grpo" else approach
    metrics = evaluate_experiment(pq_path, cfg, experiment=eval_name)
    return metrics

# ---------------------------------------------------------------------------
# Steps individuais
# ---------------------------------------------------------------------------

def step_base(cfg: dict, args: argparse.Namespace) -> dict:
    return _run_inference_and_eval(cfg, args, approach="base")

def step_few_shot(cfg: dict, args: argparse.Namespace) -> dict:
    return _run_inference_and_eval(cfg, args, approach="few_shot")

def step_sft(cfg: dict, args: argparse.Namespace) -> dict:
    from src.training.sft_trainer import run_sft
    cfg = copy.deepcopy(cfg)

    logger.info("[STEP] SFT Training (Padrão)")
    cfg["classification"]["experiment"] = "sft"
    return run_sft(cfg, resume_from_checkpoint=args.resume)

def step_sft_infer(cfg: dict, args: argparse.Namespace) -> dict:
    return _run_inference_and_eval(cfg, args, approach="sft")

def step_sft_cot(cfg: dict, args: argparse.Namespace) -> dict:
    from src.training.sft_trainer import run_sft
    cfg = copy.deepcopy(cfg)

    logger.info("[STEP] SFT-CoT Training (Fundido)")
    cfg["classification"]["experiment"] = "sft_cot"
    return run_sft(cfg, resume_from_checkpoint=args.resume)

def step_sft_cot_infer(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["classification"]["experiment"] = "sft_cot"

    # apenas para corrigir o caminho dos lora adapters
    cfg["sft"]["output_dir"] = "models/{experiment}_{mode}_{model}/lora_adapters"
    cfg["sft"]["adapter_name"] = "{experiment}_{mode}_{model}/lora_adapters"
    return _run_inference_and_eval(cfg, args, approach="sft_cot")

# --- Novas Bifurcações do GRPO ---

def step_grpo_standard(cfg: dict, args: argparse.Namespace) -> dict:
    from src.training.grpo_trainer import run_grpo
    cfg = copy.deepcopy(cfg)

    logger.info("[STEP] GRPO Training (Standard Accuracy)")
    cfg["classification"]["experiment"] = "grpo"
    cfg["grpo"]["rwfn_accuracy_mode"] = "standard"
    cfg["grpo"]["output_dir"] = "models/{experiment}_standard_{mode}_{model}"
    cfg["grpo"]["adapter_name"] = "{experiment}_standard_{mode}_{model}"
    return run_grpo(cfg, resume_from_checkpoint=args.resume)

def step_grpo_standard_infer(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)

    cfg["grpo"]["rwfn_accuracy_mode"] = "standard"
    cfg["grpo"]["output_dir"] = "models/{experiment}_standard_{mode}_{model}"
    return _run_inference_and_eval(cfg, args, approach="grpo")

def step_grpo_distance(cfg: dict, args: argparse.Namespace) -> dict:
    from src.training.grpo_trainer import run_grpo
    
    cfg = copy.deepcopy(cfg)

    logger.info("[STEP] GRPO Training (Distance Accuracy)")
    cfg["classification"]["experiment"] = "grpo"
    cfg["grpo"]["rwfn_accuracy_mode"] = "distance"
    cfg["grpo"]["output_dir"] = "models/{experiment}_distance_{mode}_{model}"
    cfg["grpo"]["adapter_name"] = "{experiment}_distance_{mode}_{model}"
    return run_grpo(cfg, resume_from_checkpoint=args.resume)

def step_grpo_distance_infer(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)

    cfg["grpo"]["rwfn_accuracy_mode"] = "distance"
    cfg["grpo"]["output_dir"] = "models/{experiment}_distance_{mode}_{model}"
    return _run_inference_and_eval(cfg, args, approach="grpo")

# ---------------------------------------------------------------------------

def step_eval(cfg: dict, args: argparse.Namespace) -> None:
    from src.evaluation.metrics import plot_all
    logger.info("[STEP] Avaliação e gráficos finais")
    
    results_dir = Path(cfg["paths"]["results_dir"])
    all_metrics = []
    
    for json_path in sorted(results_dir.glob("*_metrics.json")):
        if "consolidated" in json_path.name:
            continue
        try:
            with open(json_path) as fh:
                m = json.load(fh)
            all_metrics.append(m)
        except Exception as e:
            logger.warning("Erro ao ler %s: %s", json_path, e)

    if not all_metrics:
        logger.warning("Nenhuma métrica encontrada em %s", results_dir)
        return

    logger.info("\n%s\nRESUMO DAS MÉTRICAS\n%s", "=" * 60, "=" * 60)
    header = f"{'Experimento':<35} {'Acc':>8} {'F1-mac':>8} {'F1-wgt':>8}"
    logger.info(header)
    logger.info("-" * 60)
    for m in all_metrics:
        if "f1_macro" in m:
            logger.info(
                "%-35s %8.4f %8.4f %8.4f",
                m.get("experiment", "unknown"),
                m.get("accuracy", 0.0),
                m.get("f1_macro", 0.0),
                m.get("f1_weighted", 0.0),
            )

    consolidated = [
        {k: v for k, v in m.items() if k not in {"classification_report"}}
        for m in all_metrics
    ]
    out_json = results_dir / "consolidated_metrics.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(consolidated, fh, indent=2, ensure_ascii=False)
    logger.info("Métricas consolidadas salvas: %s", out_json)

    plot_all(cfg, all_metrics)


# ---------------------------------------------------------------------------
# Core CLI
# ---------------------------------------------------------------------------

def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)
    
    if args.mode: cfg["classification"]["mode"] = args.mode
    if args.model: cfg["model"]["name"] = args.model
    if args.batch_size: cfg["inference"]["batch_size"] = args.batch_size
    if args.seed is not None: cfg["project"]["seed"] = args.seed
    return cfg

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper2026",
        description="Pipeline de Treinamento e Avaliação - LeanDL-HPC 2025"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps", nargs="+", default=["all"], choices=ALL_STEPS + ["all"])
    parser.add_argument("--mode", type=str, choices=["3class", "2class"])
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--adapter-path", type=str, dest="adapter_path")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    cfg["logging"]["level"] = args.log_level

    setup_logger("main", log_dir=cfg["paths"]["logs_dir"], level=cfg["logging"]["level"], log_filename="main.log")
    steps = ALL_STEPS if "all" in args.steps else args.steps

    logger.info("=" * 70)
    logger.info("PAPER 2026 PIPELINE | mode=%s", cfg["classification"]["mode"])
    logger.info("=" * 70)

    step_fns = {
        "base":                step_base,
        "few_shot":            step_few_shot,
        "sft":                 step_sft,
        "sft_infer":           step_sft_infer,
        "sft_cot":             step_sft_cot,
        "sft_cot_infer":       step_sft_cot_infer,
        "grpo_standard":       step_grpo_standard,
        "grpo_standard_infer": step_grpo_standard_infer,
        "grpo_distance":       step_grpo_distance,
        "grpo_distance_infer": step_grpo_distance_infer,
        "eval":                step_eval,
    }

    t0_total = time.perf_counter()

    for step in steps:
        if step not in step_fns: continue

        t0 = time.perf_counter()
        logger.info("\n>>> Executando step: %s\n", step.upper())

        try:
            step_fns[step](cfg, args)
        except Exception as exc:
            logger.exception("Erro FATAL no step '%s': %s", step, exc)
            if not args.force:
                logger.error("Abortando pipeline.")
                sys.exit(1)

        logger.info("<<< Step '%s' concluído em %.1fs", step, time.perf_counter() - t0)

    logger.info("\n%s\nPipeline concluído com sucesso em %.1fs\n%s", "=" * 70, time.perf_counter() - t0_total, "=" * 70)

if __name__ == "__main__":
    main()