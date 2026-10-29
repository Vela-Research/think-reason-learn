"""The example notebooks run end to end, with fake models in place of real ones."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from think_reason_learn.testing import FakeLLM

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.ipynb"))
MODELS = {
    '"openai:gpt-5-mini"': '"fake:writer"',
    '"openai:gpt-5-nano"': '"fake:judge"',
    '"openai:gpt-5"': '"fake:judge"',
}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path: Path, generic_fakes: tuple[FakeLLM, FakeLLM]) -> None:
    if path.stem == "verifiable_rl":
        pytest.importorskip("torch")
    cells = json.loads(path.read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    for real, fake in MODELS.items():
        code = code.replace(real, fake)
    code = code.replace('"jev"', '"fake:judge"').replace("tree.plot()", "tree")
    exec(compile(code, str(path), "exec"), {})
