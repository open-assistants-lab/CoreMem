"""Tests for the CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_cli(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ}
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "coremem"] + args,
        capture_output=True,
        text=True,
        env=full_env,
        timeout=30,
    )


def test_cli_help():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "recall" in result.stdout
    assert "ingest" in result.stdout
    assert "mcp" in result.stdout
    assert "hook" in result.stdout


def test_cli_recall(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    _run_cli(["ingest", "user", "I love hiking in Yosemite", "--session-id", "s1"], env=env)
    result = _run_cli(["recall", "hiking", "--strategy", "direct"], env=env)
    assert result.returncode == 0
    assert "hiking" in result.stdout.lower()


def test_cli_ingest(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    result = _run_cli(["ingest", "user", "hello world", "--session-id", "s1"], env=env)
    assert result.returncode == 0
    assert "turn_id:" in result.stdout


def test_cli_sessions(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    _run_cli(["ingest", "user", "hello", "--session-id", "s1"], env=env)
    _run_cli(["ingest", "user", "world", "--session-id", "s2"], env=env)
    result = _run_cli(["sessions"], env=env)
    assert result.returncode == 0
    assert "s1" in result.stdout
    assert "s2" in result.stdout


def test_cli_hook_reads_stdin(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    hook_input = json.dumps({"prompt": "test prompt", "session_id": "s1"})
    result = subprocess.run(
        [sys.executable, "-m", "coremem", "hook", "user_prompt_submit"],
        input=hook_input,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=30,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert "hookSpecificOutput" in output