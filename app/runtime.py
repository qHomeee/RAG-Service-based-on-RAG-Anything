import logging
import os

from app.config import settings


logger = logging.getLogger("rag_service")
_configured = False


def configure_cpu_runtime() -> None:
    """Bound per-worker CPU usage before ML models are loaded."""
    global _configured
    if _configured:
        return

    threads = max(1, settings.cpu_threads_per_worker)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, settings.cpu_interop_threads))
        except RuntimeError:
            # PyTorch only allows changing inter-op threads before parallel work starts.
            pass
    except Exception as exc:
        logger.warning("cpu_runtime_torch_configuration_skipped", extra={"error": str(exc)})

    logger.info(
        "cpu_runtime_configured",
        extra={
            "threads_per_worker": threads,
            "interop_threads": max(1, settings.cpu_interop_threads),
            "uvicorn_workers": settings.uvicorn_workers,
        },
    )
    _configured = True
