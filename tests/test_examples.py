"""The example notebooks run end to end, with fake models in place of real ones."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from think_reason_learn.llm import Question
from think_reason_learn.testing import FakeLLM, GenerateRequest, fake_llms

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.ipynb"))
MODELS = {'"openai:gpt-5-mini"': '"fake:writer"', '"openai:gpt-5"': '"fake:judge"'}


def _write(request: GenerateRequest) -> dict[str, object] | str:
    fields = request.schema.model_fields if request.schema is not None else {}
    if "questions" in fields:
        texts = ["Has the founder sold a company?", "Is the founder an engineer?"]
        plain = fields["questions"].annotation == list[str]
        return {"questions": texts if plain else [{"text": t} for t in texts]}
    if "rules" in fields:
        return {"rules": ["Founders with exits succeed.", "Engineers succeed."]}
    if "rule" in fields:
        return {"reasoning": "Exits matter.", "rule": "IF exit THEN successful"}
    return "Look for prior exits and engineering depth."


def _judge(sample: str, question: Question) -> float:
    return 0.9 if any(w in sample for w in ("exit", "engineer", "sold")) else 0.1


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path: Path) -> None:
    if path.stem == "verifiable_rl":
        pytest.importorskip("torch")
    cells = json.loads(path.read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    for real, fake in MODELS.items():
        code = code.replace(real, fake)
    code = code.replace('"jev"', '"fake:judge"').replace("tree.plot()", "tree")
    with fake_llms(
        writer=FakeLLM(generate=_write),
        judge=FakeLLM(answer=_judge, kind="decision"),
    ):
        exec(compile(code, str(path), "exec"), {})
