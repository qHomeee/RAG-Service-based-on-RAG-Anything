import sys
import types

from app.mineru_offline_cli import main
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

    rc = main()
    assert rc == 0
    assert "mineru.utils.llm_aided" in sys.modules
    assert sys.modules["mineru.utils.llm_aided"].llm_aided_title("abc") == "abc"
