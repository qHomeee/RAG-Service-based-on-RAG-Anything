from __future__ import annotations

import os
import runpy
import sys
import traceback
import types


def _install_llm_aided_stub() -> None:
    disable_llm = os.getenv("DISABLE_MINERU_LLM", "1") == "1"
    if not disable_llm:
        return

    # Install stub before any mineru.* imports so openai-backed llm_aided is never required.
    stub = types.ModuleType("mineru.utils.llm_aided")

    def llm_aided_title(title: str) -> str:
        return title

    stub.llm_aided_title = llm_aided_title
    sys.modules["mineru.utils.llm_aided"] = stub


def _print_transformers_hint() -> None:
    print(
        "В .venv-mineru несовместим transformers. "
        "Попробуй: pip install -U 'transformers==4.35.0' 'huggingface_hub==0.17.3'",
        file=sys.stderr,
    )


def _is_transformers_compat_error(exc: BaseException) -> bool:
    text = str(exc)
    lowered = text.lower()
    return "transformers" in lowered or "find_pruneable_heads_and_indices" in text


def main() -> int:
    _install_llm_aided_stub()

    try:
        try:
            from mineru.cli.client import main as mineru_main

            # Some MinerU versions expose click command as module-level main.
            try:
                mineru_main(standalone_mode=False)
            except TypeError:
                # Older variants may define main() without click-style kwargs.
                mineru_main()
        except (ImportError, AttributeError):
            # Fallback for versions where entrypoint shape differs.
            runpy.run_module("mineru.cli.client", run_name="__main__")
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except ImportError as exc:
        if _is_transformers_compat_error(exc):
            _print_transformers_hint()
            return 1
        traceback.print_exc()
        return 1
    except Exception as exc:  # pragma: no cover
        if _is_transformers_compat_error(exc):
            _print_transformers_hint()
            return 1
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
