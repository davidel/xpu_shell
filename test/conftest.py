import os
import tempfile
import pytest


@pytest.fixture(autouse=True)
def isolate_xpu_shell_config(monkeypatch):
  with tempfile.TemporaryDirectory() as td:
    monkeypatch.setenv("XPU_SHELL_CONFIG_DIR", td)
    monkeypatch.setenv("RUNPOD_SHELL_CONFIG_DIR", td)
    yield td
