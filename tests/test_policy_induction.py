"""Tests for PolicyInduction, focused on batched LLM calls."""

from __future__ import annotations

import math

import numpy as np
import orjson
import pandas as pd
import pytest

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.policy_induction import PolicyInduction
from think_reason_learn.policy_induction._policy_induction import Policies
from tests.fake_policy_llm import FakePolicyLLM


TASK = "Predict whether a startup founder will succeed based on their background."

X = pd.DataFrame(
    {
        "founder_info": [
            "Alex is a serial entrepreneur with two successful exits and AI expertise.",
            "Jordan graduated top of class from MIT but has no business experience.",
            "Taylor has 10 years in finance and secured seed funding quickly.",
            "Casey started a company out of high school and faced multiple failures.",
            "Morgan is a former Google engineer with machine learning patents.",
            "Riley has deep domain knowledge but has never managed a team.",
            "Sam built and sold a marketplace business to a public company.",
            "Quinn is a first-time founder working alone without funding.",
        ]
    }
)
Y = ["YES", "NO", "YES", "NO", "YES", "NO", "YES", "NO"]


def make_pi(tmp_path, fake: FakePolicyLLM, **kw) -> PolicyInduction:
    """A PolicyInduction wired to the fake, with a task already set."""
    kw.setdefault("policy_batch_size", 3)
    kw.setdefault("llm_semaphore_limit", 3)
    pi = PolicyInduction(
        gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")],
        confirm_requests=False,
        save_path=tmp_path,
        name="test_pi",
        max_samples_as_context=4,
        config={"cv_folds": 2, "Cs": (1.0,)},
        _llm=fake,
        **kw,
    )
    pi._task_description = TASK
    pi._pgen_instructions_template = "Enrich policies. Max <max_policy_length>."
    return pi


def seed_for_scoring(pi: PolicyInduction, fake: FakePolicyLLM) -> None:
    """Populate data + policy memory so _score_policies can run standalone."""
    pi._set_data(X, Y)
    pi._policy_memory = pd.DataFrame(
        {
            "policy": fake.policy_texts(),
            "predictions": [None] * fake.n_policies,
        }
    )


def null_cells(pi: PolicyInduction) -> int:
    preds = pi._policy_memory["predictions"]
    return int(sum(int(s.isna().sum()) for s in preds if s is not None))


# ── Batching arithmetic ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_call_count(tmp_path):
    """One call per (sample, policy-chunk), and no single-policy calls."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, policy_batch_size=3)
    seed_for_scoring(pi, fake)

    await pi._score_policies()

    assert fake.batch_calls == len(X) * math.ceil(6 / 3) == 16
    assert fake.single_calls == 0
    assert null_cells(pi) == 0


@pytest.mark.asyncio
async def test_batch_size_one_matches_unbatched(tmp_path):
    """policy_batch_size=1 uses the single-item path and yields the same matrix."""
    fake_b = FakePolicyLLM(n_policies=6)
    pi_b = make_pi(tmp_path / "batched", fake_b, policy_batch_size=3)
    seed_for_scoring(pi_b, fake_b)
    await pi_b._score_policies()

    fake_s = FakePolicyLLM(n_policies=6)
    pi_s = make_pi(tmp_path / "single", fake_s, policy_batch_size=1)
    seed_for_scoring(pi_s, fake_s)
    await pi_s._score_policies()

    assert fake_s.batch_calls == 0
    assert fake_s.single_calls == len(X) * 6 == 48

    np.testing.assert_array_equal(
        pi_b._build_feature_matrix()[0], pi_s._build_feature_matrix()[0]
    )


@pytest.mark.parametrize("bs", [1, 2, 10])
def test_estimate_fit_requests_batched(tmp_path, bs):
    """Scoring estimate is ceil(missing policies / batch) summed over samples."""
    fake = FakePolicyLLM(n_policies=5)
    pi = make_pi(tmp_path, fake, policy_batch_size=bs)
    seed_for_scoring(pi, fake)

    # Policy 0 fully scored, policy 1 half scored, policies 2-4 absent entirely.
    sample_keys = [str(i) for i in X.index]
    scores = {
        "0": {k: "YES" for k in sample_keys},
        "1": {k: (None if i % 2 else "NO") for i, k in enumerate(sample_keys)},
    }
    pi._write_ckpt(
        pi._FIT_CKPT_NAME, {"policies": fake.policy_texts(), "scores": scores}
    )

    # Per sample: 3 absent policies, plus policy 1 on odd-indexed samples.
    expected = sum(
        math.ceil((3 + (1 if i % 2 else 0)) / bs) for i in range(len(sample_keys))
    )
    key = next(k for k in pi._estimate_fit_requests() if "scoring" in k)
    assert pi._estimate_fit_requests()[key] == expected


@pytest.mark.parametrize("bs", [1, 3, 10])
def test_estimate_predict_requests_batched(tmp_path, bs):
    """Predict estimate batches over the nonzero-weight policies only."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, policy_batch_size=bs)
    pi._feature_order_ = np.array([str(i) for i in range(6)], dtype=str)

    class _LR:
        coef_ = np.array([[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]])

    pi._lr = _LR()  # type: ignore[assignment]

    key = next(k for k in pi._estimate_predict_requests(X) if "predict" in k)
    assert pi._estimate_predict_requests(X)[key] == len(X) * math.ceil(3 / bs)


# ── Generation batch cap ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_gen_batches_caps_generation(tmp_path):
    """max_gen_batches stops generation early even though both classes remain."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, max_gen_batches=1)
    pi._set_data(X, Y)

    policies = await pi._run_generation(pi._get_gen_instructions())

    assert fake.count_of(Policies) == 1
    assert policies == fake.policy_texts()
    ckpt = pi._read_ckpt(pi._FIT_CKPT_NAME)
    assert ckpt is not None and ckpt["batches_done"] == 1


@pytest.mark.asyncio
async def test_max_gen_batches_none_runs_until_class_exhaustion(tmp_path):
    """None disables the cap: generation stops only once a class runs out."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, max_gen_batches=None)
    pi._set_data(X, Y)

    await pi._run_generation(pi._get_gen_instructions())

    # 8 rows, 4 YES / 4 NO, batches of 4 (2 YES + 2 NO) => exhausts after 2.
    assert fake.count_of(Policies) == 2


@pytest.mark.parametrize("cap", [1, 2, 5])
def test_estimate_fit_requests_respects_max_gen_batches(tmp_path, cap):
    """The generation estimate is capped the same way _run_generation is."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, max_gen_batches=cap)
    pi._set_data(X, Y)

    key = next(k for k in pi._estimate_fit_requests() if "generation" in k)
    assert pi._estimate_fit_requests()[key] == min(cap, 2)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1", True])
def test_validate_init_rejects_bad_max_gen_batches(bad):
    with pytest.raises(ValueError, match="max_gen_batches"):
        PolicyInduction(
            gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")],
            max_gen_batches=bad,
        )


def test_max_gen_batches_round_trips(tmp_path):
    """The saved cap is restored on load."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, max_gen_batches=1)
    seed_for_scoring(pi, fake)
    pi.save(tmp_path / "model")

    assert PolicyInduction.load(tmp_path / "model").max_gen_batches == 1


def test_max_gen_batches_default_round_trips(tmp_path):
    """Generation is capped at 7 batches by default, and that survives a save."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake)  # max_gen_batches left at its default
    assert pi.max_gen_batches == 7
    seed_for_scoring(pi, fake)
    pi.save(tmp_path / "model")

    manifest = orjson.loads((tmp_path / "model" / "policy_induction.json").read_bytes())
    assert manifest["max_gen_batches"] == 7
    assert PolicyInduction.load(tmp_path / "model").max_gen_batches == 7


def test_max_gen_batches_explicit_none_round_trips(tmp_path):
    """An explicit None (uncapped) is preserved, not coerced to the default."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, max_gen_batches=None)
    seed_for_scoring(pi, fake)
    pi.save(tmp_path / "model")

    manifest = orjson.loads((tmp_path / "model" / "policy_induction.json").read_bytes())
    assert manifest["max_gen_batches"] is None
    assert PolicyInduction.load(tmp_path / "model").max_gen_batches is None


def test_load_manifest_without_max_gen_batches(tmp_path):
    """A save predating the key loads uncapped, matching how it was trained."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake)
    seed_for_scoring(pi, fake)
    pi.save(tmp_path / "model")

    path = tmp_path / "model" / "policy_induction.json"
    manifest = orjson.loads(path.read_bytes())
    del manifest["max_gen_batches"]
    path.write_bytes(orjson.dumps(manifest))

    assert PolicyInduction.load(tmp_path / "model").max_gen_batches is None


# ── Alignment and recovery ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_id_misalignment_requeries_individually(tmp_path):
    """Shifted ids resolve to nothing, so the whole chunk is re-queried."""
    fake = FakePolicyLLM(n_policies=6, shift_ids=100, shift_first_only=True)
    pi = make_pi(tmp_path, fake, policy_batch_size=3)
    seed_for_scoring(pi, fake)

    await pi._score_policies()

    assert fake.single_calls == 3  # exactly the first chunk
    assert null_cells(pi) == 0


@pytest.mark.asyncio
async def test_partial_batch_requeries_only_missing(tmp_path):
    """A dropped id costs exactly one extra single call per batch."""
    fake = FakePolicyLLM(n_policies=6, drop_ids={2})
    pi = make_pi(tmp_path, fake, policy_batch_size=3)
    seed_for_scoring(pi, fake)

    await pi._score_policies()

    assert fake.batch_calls == 16
    assert fake.single_calls == 16
    assert null_cells(pi) == 0


@pytest.mark.asyncio
async def test_persistent_miss_leaves_cells_none(tmp_path):
    """When the re-query also fails, cells stay unscored rather than wrong."""
    fake = FakePolicyLLM(n_policies=6, drop_ids={2}, fail_single=True)
    pi = make_pi(tmp_path, fake, policy_batch_size=3)
    seed_for_scoring(pi, fake)

    await pi._score_policies()  # must not raise

    # Local id 2 of each chunk = policies 2 and 5.
    preds = pi._policy_memory["predictions"]
    assert preds.at[2].isna().all() and preds.at[5].isna().all()
    for pid in (0, 1, 3, 4):
        assert not preds.at[pid].isna().any()


@pytest.mark.asyncio
async def test_batch_failure_does_not_fan_out(tmp_path):
    """A failed batch call leaves cells unscored without N individual retries."""
    fake = FakePolicyLLM(n_policies=6, fail_after=1)
    pi = make_pi(tmp_path, fake, policy_batch_size=3)
    seed_for_scoring(pi, fake)

    await pi._score_policies()

    assert fake.single_calls == 0
    assert null_cells(pi) == len(X) * 6


# ── Checkpointing ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_resume_across_batch_boundary(tmp_path):
    """A resumed run re-chunks only the cells that are still missing."""
    fake = FakePolicyLLM(n_policies=6, fail_after=9)
    pi = make_pi(tmp_path, fake, policy_batch_size=3, llm_semaphore_limit=1)
    seed_for_scoring(pi, fake)

    await pi._score_policies()

    # 16 units in order, 2 per sample; calls 9+ fail => samples 4-7 unscored.
    ckpt = pi._read_ckpt(pi._FIT_CKPT_NAME)
    assert ckpt is not None
    scores = ckpt["scores"]
    for pid in range(6):
        row = scores[str(pid)]
        assert all(row[str(s)] is not None for s in range(4))
        assert all(row[str(s)] is None for s in range(4, 8))

    # Resume on a fresh instance pointed at the same save_path.
    fake2 = FakePolicyLLM(n_policies=6)
    pi2 = make_pi(tmp_path, fake2, policy_batch_size=3, llm_semaphore_limit=1)
    seed_for_scoring(pi2, fake2)

    await pi2._score_policies()

    assert fake2.batch_calls == 4 * math.ceil(6 / 3) == 8
    assert null_cells(pi2) == 0


# ── Persistence ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_policy_batch_size_round_trips(tmp_path):
    """The saved batch size is restored, keeping scoring and predict in sync."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, policy_batch_size=4)
    seed_for_scoring(pi, fake)
    await pi._score_policies()
    pi.save(tmp_path / "model")

    manifest = orjson.loads((tmp_path / "model" / "policy_induction.json").read_bytes())
    assert manifest["version"] == 3
    assert manifest["policy_batch_size"] == 4
    assert PolicyInduction.load(tmp_path / "model").policy_batch_size == 4


def test_load_tolerates_null_policy_batch_size(tmp_path):
    """A present-but-None key must fall back to the default, never become None."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, policy_batch_size=4)
    seed_for_scoring(pi, fake)
    pi.save(tmp_path / "model")

    path = tmp_path / "model" / "policy_induction.json"
    manifest = orjson.loads(path.read_bytes())
    manifest["policy_batch_size"] = None
    path.write_bytes(orjson.dumps(manifest))

    assert PolicyInduction.load(tmp_path / "model").policy_batch_size == 10


@pytest.mark.parametrize("bad", [0, -1, 51, 3.0, "3", None, True])
def test_validate_init_rejects_bad_batch_size(bad):
    with pytest.raises(ValueError, match="policy_batch_size"):
        PolicyInduction(
            gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")],
            policy_batch_size=bad,
        )


# ── Predict ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fit_and_predict_batched(tmp_path):
    """End-to-end fit, then predict batching over nonzero-weight policies."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, policy_batch_size=3)

    await pi.fit(X, Y)
    assert not (tmp_path / pi._FIT_CKPT_NAME).exists()

    n_used = int(np.count_nonzero(pi.lr.coef_[0]))
    fake.reset()
    records = [rec async for rec in pi.predict(X)]

    assert len(records) == len(X)
    assert fake.batch_calls == len(X) * math.ceil(n_used / 3)


@pytest.mark.asyncio
async def test_predict_skips_sample_on_total_miss(tmp_path):
    """No answers at all means no record and a retained checkpoint."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, policy_batch_size=3)
    await pi.fit(X, Y)

    fake.fail_after = 1
    fake.fail_single = True
    records = [rec async for rec in pi.predict(X)]

    assert records == []
    assert (tmp_path / pi._PREDICT_CKPT_NAME).exists()


@pytest.mark.asyncio
async def test_predict_warns_on_partial_miss(tmp_path, caplog):
    """A partial miss still yields, but says so instead of silently scoring 0."""
    fake = FakePolicyLLM(n_policies=6)
    pi = make_pi(tmp_path, fake, policy_batch_size=3)
    await pi.fit(X, Y)
    if int(np.count_nonzero(pi.lr.coef_[0])) < 2:
        pytest.skip("model kept too few policies to exercise a partial miss")

    fake.drop_ids = {0}
    fake.fail_single = True
    with caplog.at_level("WARNING"):
        records = [rec async for rec in pi.predict(X)]

    assert len(records) == len(X)
    assert any("answers missing after re-query" in m for m in caplog.messages)
