"""
src/utils/logger.py
Setup de logging estruturado com saída para console e arquivo .log.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGERS: dict[str, logging.Logger] = {}


def setup_logger(
    name: str,
    log_dir: str | Path,
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """
    Cria e configura um logger com dois handlers:
      - StreamHandler (stdout)
      - FileHandler  (log_dir/<name>.log  ou log_dir/<log_filename>)

    Idempotente: se o logger já foi configurado, retorna a instância existente.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    filename = log_filename or f"{name.replace('/', '_').replace('.', '_')}.log"
    log_file = log_dir / filename

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger já configurado, ou cria um básico se inexistente."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    return logging.getLogger(name)