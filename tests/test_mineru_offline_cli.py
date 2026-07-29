import sys
import types

from app.mineru_offline_cli import (
    _guarded_para_text_merge,
    _guarded_text_block_merge,
    main,
)
from app.mineru_runner import build_mineru_command


def test_build_command_uses_offline_entrypoint(tmp_path):
    cmd = build_mineru_command(tmp_path / "python", tmp_path / "a.pdf", tmp_path / "out")
    assert "app.mineru_offline_cli" in cmd


def test_offline_cli_installs_llm_stub(monkeypatch):
    fake_mineru = types.ModuleType("mineru")
    fake_cli_pkg = types.ModuleType("mineru.cli")
    fake_client = types.ModuleType("mineru.cli.client")

    def _main():
        return None

    fake_client.main = _main
    fake_cli_pkg.client = fake_client
    fake_mineru.cli = fake_cli_pkg
    monkeypatch.setitem(sys.modules, "mineru", fake_mineru)
    monkeypatch.setitem(sys.modules, "mineru.cli", fake_cli_pkg)
    monkeypatch.setitem(sys.modules, "mineru.cli.client", fake_client)
    monkeypatch.setenv("DISABLE_MINERU_LLM", "1")

    rc = main(["--path", "in.pdf", "--output", "out"])
    assert rc == 0
    assert "mineru.utils.llm_aided" in sys.modules
    assert sys.modules["mineru.utils.llm_aided"].llm_aided_title("abc") == "abc"


def test_missing_inline_equation_content_skips_unsafe_mineru_merge():
    calls = []

    def original(block1, block2):
        calls.append((block1, block2))
        return "merged"

    guarded = _guarded_text_block_merge(original)
    next_block = {"lines": [{"spans": [{"content": "Продолжение"}]}]}
    equation_block = {
        "lines": [
            {
                "spans": [
                    {
                        "type": "inline_equation",
                        "image_path": "formula.jpg",
                    }
                ]
            }
        ]
    }

    assert guarded(next_block, equation_block) == (next_block, equation_block)
    assert calls == []


def test_normal_text_spans_still_use_native_mineru_merge():
    def original(block1, block2):
        return block1["id"], block2["id"]

    guarded = _guarded_text_block_merge(original)
    next_block = {
        "id": "next",
        "lines": [{"spans": [{"content": "продолжение"}]}],
    }
    previous_block = {
        "id": "previous",
        "lines": [{"spans": [{"content": "текст"}]}],
    }

    assert guarded(next_block, previous_block) == ("next", "previous")


def test_missing_inline_equation_content_is_safe_for_markdown_merge():
    captured = []

    def original(para_block):
        captured.append(para_block)
        return para_block["lines"][0]["spans"][0]["content"]

    guarded = _guarded_para_text_merge(original)
    equation_block = {
        "type": "text",
        "lines": [
            {
                "spans": [
                    {
                        "type": "inline_equation",
                        "image_path": "formula.jpg",
                    }
                ]
            }
        ],
    }

    assert guarded(equation_block) == ""
    assert captured[0]["lines"][0]["spans"][0]["content"] == ""


def test_markdown_merge_preserves_existing_span_content():
    def original(para_block):
        return para_block["lines"][0]["spans"][0]["content"]

    guarded = _guarded_para_text_merge(original)
    text_block = {
        "lines": [{"spans": [{"type": "text", "content": "NaCl"}]}],
    }

    assert guarded(text_block) == "NaCl"
