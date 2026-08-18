"""
src/utils/resource_monitor.py
Monitoramento de pico de VRAM (GPU), RAM (CPU), energia e emissões de carbono.

Estratégia:
  - VRAM : torch.cuda.max_memory_allocated() (primário) + pynvml (fallback)
  - RAM  : psutil.Process().memory_info().rss / resource (pico real no Linux)
  - CO2  : codecarbon (EmissionsTracker)
  - Decorador @monitor_resources para capturar automaticamente.
"""

import functools
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional
import threading

import psutil

# Tenta importar resource para medição de pico de RAM nativa no Linux
try:
    import resource
    _RESOURCE_AVAILABLE = True
except ImportError:
    _RESOURCE_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False

# Importação segura do CodeCarbon
try:
    from codecarbon import EmissionsTracker
    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class ResourceStats:
    """Estatísticas de uso de recursos de um experimento."""
    experiment: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_seconds: float = 0.0
    n_samples: int = 0
    throughput_samples_per_sec: float = 0.0

    # Memória
    peak_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    
    # Sustentabilidade (CodeCarbon)
    emissions_kg_co2: float = 0.0

    # Extras livres
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    def log_summary(self, logger=None) -> None:
        msg = (
            f"[ResourceStats] {self.experiment} | "
            f"time={self.elapsed_seconds:.1f}s | "
            f"vram={self.peak_vram_mb:.1f}MB | "
            f"ram={self.peak_ram_mb:.1f}MB | "
            f"co2={self.emissions_kg_co2:.6f}kg"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """
    Contexto para medir VRAM, RAM, Tempo, e Pegada de Carbono.
    """

    def __init__(self, experiment: str = "experiment", n_samples: int = 0):
        self.experiment = experiment
        self.n_samples = n_samples
        self.stats: Optional[ResourceStats] = None
        self._process = psutil.Process()
        self._tracker = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResourceMonitor":
        self._ram_baseline_mb = self._current_ram_mb()
        self._peak_ram_during = self._ram_baseline_mb
        self._stop_polling = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_ram, daemon=True)
        self._poll_thread.start()

        if _TORCH_AVAILABLE and torch.cuda.is_available():
            # Não reseta — captura baseline atual (modelo já carregado)
            self._vram_baseline_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()  # reseta só para pegar o pico a partir daqui
        else:
            self._vram_baseline_mb = 0.0
        
        # Inicia medição de RAM no Linux
        if _RESOURCE_AVAILABLE:
            self._start_ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        else:
            self._start_ram = self._current_ram_mb()

        # Inicia o CodeCarbon silenciosamente (log_level="error" para não poluir seus logs)
        if _CODECARBON_AVAILABLE:
            # O tracker salva um emissions.csv automático na raiz do projeto
            self._tracker = EmissionsTracker(
                project_name=self.experiment, 
                log_level="error"
            )
            self._tracker.start()

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self._stop_polling.set()
        self._poll_thread.join()
        peak_ram = self._peak_ram_during - self._ram_baseline_mb

        end_time = time.perf_counter()
        elapsed = end_time - self._start_time

        peak_vram = self._vram_baseline_mb + self._peak_vram_mb()

        throughput = self.n_samples / elapsed if elapsed > 0 and self.n_samples > 0 else 0.0

        # Para o CodeCarbon e captura os dados
        emissions = 0.0
        if self._tracker is not None:
            emissions = self._tracker.stop()
            # O CodeCarbon guarda o consumo de CPU + GPU + RAM

        self.stats = ResourceStats(
            experiment=self.experiment,
            start_time=self._start_time,
            end_time=end_time,
            elapsed_seconds=round(elapsed, 3),
            n_samples=self.n_samples,
            throughput_samples_per_sec=round(throughput, 4),
            peak_vram_mb=round(peak_vram, 2),
            peak_ram_mb=round(peak_ram, 2),
            emissions_kg_co2=emissions
        )

    # ------------------------------------------------------------------
    # Decorador estático
    # ------------------------------------------------------------------

    @staticmethod
    def wrap(experiment: str = "experiment", n_samples_kwarg: str = "n_samples"):
        """Decorador que injeta monitoramento em qualquer função."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                n = kwargs.get(n_samples_kwarg, 0)
                with ResourceMonitor(experiment, n_samples=n) as mon:
                    result = fn(*args, **kwargs)
                mon.stats.log_summary()
                return result, mon.stats
            return wrapper
        return decorator

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _poll_ram(self):
        while not self._stop_polling.is_set():
            current = self._current_ram_mb()
            if current > self._peak_ram_during:
                self._peak_ram_during = current
            self._stop_polling.wait(timeout=1.0)

    def _reset_vram(self) -> None:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _peak_vram_mb(self) -> float:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        if _PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return info.used / (1024 ** 2)
            except Exception:
                pass
        return 0.0

    def _current_ram_mb(self) -> float:
        return self._process.memory_info().rss / (1024 ** 2)