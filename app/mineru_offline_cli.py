from __future__ import annotations

import argparse
import os
import importlib
import importlib.metadata
import inspect
import runpy
import sys
import traceback
import types
from contextlib import contextmanager

# PyTorch >= 2.6 defaults torch.load(..., weights_only=True), which can break
# YOLOv10 checkpoint loading used by doclayout_yolo during MinerU model init.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline-safe MinerU runner")
    parser.add_argument("--path", help="Input file or directory path")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("-b", "--backend", default="pipeline")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("--offline", dest="offline", action="store_true", default=True)
    parser.add_argument("--no-offline", dest="offline", action="store_false")
    parser.add_argument("--doctor", action="store_true", help="Check key MinerU runtime imports")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _install_llm_aided_stub() -> None:
    # Default is offline mode: disable llm_aided integration unless user explicitly opts out.
    stub = types.ModuleType("mineru.utils.llm_aided")

    def llm_aided_title(title: str) -> str:
        return title

    stub.llm_aided_title = llm_aided_title
    sys.modules["mineru.utils.llm_aided"] = stub


def _edge_span(block: object, *, first: bool) -> dict | None:
    if not isinstance(block, dict):
        return None
    lines = block.get("lines")
    if not isinstance(lines, list) or not lines:
        return None
    line = lines[0] if first else lines[-1]
    if not isinstance(line, dict):
        return None
    spans = line.get("spans")
    if not isinstance(spans, list) or not spans:
        return None
    span = spans[0] if first else spans[-1]
    return span if isinstance(span, dict) else None


def _guarded_text_block_merge(original):
    def guarded(block1, block2):
        first_span = _edge_span(block1, first=True)
        last_span = _edge_span(block2, first=False)
        if (
            first_span is not None
            and not isinstance(first_span.get("content"), str)
        ) or (
            last_span is not None
            and not isinstance(last_span.get("content"), str)
        ):
            return block1, block2
        return original(block1, block2)

    guarded._rag_missing_span_content_guard = True
    return guarded


def _guarded_para_text_merge(original):
    def guarded(para_block):
        if isinstance(para_block, dict):
            lines = para_block.get("lines")
            if isinstance(lines, list):
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    spans = line.get("spans")
                    if not isinstance(spans, list):
                        continue
                    for span in spans:
                        if (
                            isinstance(span, dict)
                            and not isinstance(span.get("content"), str)
                        ):
                            # MinerU 2.6 can emit image-backed inline equations
                            # without `content`, while its Markdown renderer
                            # indexes the key unconditionally. Formula parsing is
                            # disabled in the CPU profile, so an empty textual
                            # representation is the safe loss-bounded fallback.
                            span["content"] = ""
        return original(para_block)

    guarded._rag_missing_span_content_guard = True
    return guarded


def _install_missing_span_content_guard() -> None:
    try:
        para_split_module = importlib.import_module(
            "mineru.backend.pipeline.para_split"
        )
        current = getattr(para_split_module, "__merge_2_text_blocks")
    except (ImportError, AttributeError):
        pass
    else:
        if not getattr(current, "_rag_missing_span_content_guard", False):
            setattr(
                para_split_module,
                "__merge_2_text_blocks",
                _guarded_text_block_merge(current),
            )

    try:
        markdown_module = importlib.import_module(
            "mineru.backend.pipeline.pipeline_middle_json_mkcontent"
        )
        current_markdown_merge = getattr(markdown_module, "merge_para_with_text")
    except (ImportError, AttributeError):
        pass
    else:
        if not getattr(
            current_markdown_merge,
            "_rag_missing_span_content_guard",
            False,
        ):
            setattr(
                markdown_module,
                "merge_para_with_text",
                _guarded_para_text_merge(current_markdown_merge),
            )


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


def _doctor() -> int:
    modules = ["mineru", "transformers", "torch", "rapid_table"]
    failed = False
    print("[mineru-doctor] checking imports...")
    for name in modules:
        try:
            importlib.import_module(name)
            try:
                ver = importlib.metadata.version(name.replace("_", "-"))
            except Exception:
                ver = "unknown"
            print(f"  OK  {name} (version={ver})")
        except Exception as exc:
            failed = True
            print(f"  FAIL {name}: {exc}")
    return 1 if failed else 0


@contextmanager
def _patched_argv(args: list[str]):
    old = sys.argv[:]
    sys.argv = ["mineru.cli.client", *args]
    try:
        yield
    finally:
        sys.argv = old


def _call_mineru_main(mineru_main, args: list[str]) -> None:
    call_attempts = [
        lambda: mineru_main(args=args, standalone_mode=False),
        lambda: mineru_main(standalone_mode=False),
        lambda: mineru_main(args),
        lambda: mineru_main(),
    ]
    for attempt in call_attempts:
        try:
            with _patched_argv(args):
                attempt()
            return
        except TypeError:
            continue
    with _patched_argv(args):
        mineru_main()


def _invoke_mineru(mineru_args: list[str]) -> None:
    try:
        import mineru.cli.client as mineru_client

        if hasattr(mineru_client, "main"):
            # `mineru.cli.client.cli` is not exported in some MinerU versions; use module-level main.
            _call_mineru_main(mineru_client.main, mineru_args)
            return
    except ImportError:
        raise

    # Fallback for versions with different entrypoint layout.
    with _patched_argv(mineru_args):
        runpy.run_module("mineru.cli.client", run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else argv
    parser = _build_parser()
    args, passthrough = parser.parse_known_args(argv)

    if args.doctor:
        return _doctor()

    if args.offline:
        _install_llm_aided_stub()
        _install_missing_span_content_guard()

    if not args.path or not args.output:
        parser.error("--path and --output are required unless --doctor is used")

    mineru_args = [
        "--path",
        args.path,
        "--output",
        args.output,
        "-b",
        args.backend,
        "-d",
        args.device,
        *passthrough,
    ]

    if args.offline:
        # Force lightweight text mode in offline mode to avoid LLM-aided/title-enhancement path.
        if "-m" not in mineru_args and "--mode" not in mineru_args:
            mineru_args += ["-m", "txt"]
        if "-f" not in mineru_args and "--formula" not in mineru_args:
            mineru_args += ["-f", "false"]
        if "-t" not in mineru_args and "--table" not in mineru_args:
            mineru_args += ["-t", "false"]

    try:
        _invoke_mineru(mineru_args)
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except ImportError as exc:
        if _is_transformers_compat_error(exc):
            _print_transformers_hint()
            return 1
        print(f"MinerU import error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    except Exception as exc:  # pragma: no cover
        if _is_transformers_compat_error(exc):
            _print_transformers_hint()
            return 1
        print(f"MinerU pipeline failed: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# PowerShell example:
# python -m app.mineru_offline_cli --path "storage/raw/book.pdf" --output "storage/parsed/book" -b pipeline -d cpu --offline
