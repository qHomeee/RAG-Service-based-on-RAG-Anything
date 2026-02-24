from __future__ import annotations

import os
import sys
import types


def _install_llm_aided_stub() -> None:
    disable_llm = os.getenv("DISABLE_MINERU_LLM", "1") == "1"
    if not disable_llm:
        try:
            from mineru.utils.llm_aided import llm_aided_title  # noqa: F401
            return
        except Exception:
            pass

    stub = types.ModuleType("mineru.utils.llm_aided")

    def llm_aided_title(title: str) -> str:
        return title

    stub.llm_aided_title = llm_aided_title
    sys.modules["mineru.utils.llm_aided"] = stub


def main() -> int:
    _install_llm_aided_stub()
    from mineru.cli.client import cli

    try:
        cli.main(args=sys.argv[1:], standalone_mode=False)
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code


if __name__ == "__main__":
    raise SystemExit(main())
