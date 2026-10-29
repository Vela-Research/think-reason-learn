# Think Reason Learn
[Website](https://thinkreasonlearn.com/) · [Docs](https://thinkreasonlearn.com/docs/)

TRL is an open-source Python library that extends classical machine learning with language-based reasoning. It
keeps the shape of decision trees, rule forests and short policies, and lets a language model reason at each step,
so every prediction comes with the questions or rules behind it. Think of it as scikit-learn with reasoning built in:
every model is a scikit-learn classifier.

TRL is developed by Vela Research with the University of Oxford. Our first applications are in venture capital, but
the library is domain-agnostic: the [phishing example](examples/phishing.ipynb) learns which questions flag
phishing emails, with one OpenAI key.

## Think, reason, learn

Every model splits the work three ways:

| | Setting | What it does |
|---|---|---|
| **Think** | `generator=` | A reasoning model writes the questions, rules or policy. |
| **Reason** | `answerer=` | A model answers them for each sample: a reasoning model, or a decision model such as Jev that returns probabilities. |
| **Learn** | `fit()` | The model learns which answers predict the label. |

## Models

- **GPTree**: a decision tree whose splits are questions written by a language model.
- **RRF (Random Rule Forest)**: YES/NO questions, weighed by elastic net (or put to the paper's vote).
- **Policy Induction**: a short policy of rules refined batch by batch, its rules weighed to predict.
- **Reasoned Rule Mining**: rules mined from reasoning, applied in a cheap-then-strict cascade.
- **VerifiableRL**: learns which information to reveal, one slot at a time, at a cost.

The papers are on our [research page](https://thinkreasonlearn.com/research).

## Install

```bash
pip install think-reason-learn             # OpenAI-compatible providers and Jev
pip install 'think-reason-learn[anthropic]' # + Claude
pip install 'think-reason-learn[rl]'        # + PyTorch, for VerifiableRL
pip install 'think-reason-learn[all]'
```

Python 3.13+. Drawing trees needs [Graphviz](https://graphviz.org/download/).

## Quick start

```python
from think_reason_learn.gptree import GPTree

X = [
    "Alex is a serial entrepreneur with two exits and a strong network in AI.",
    "Jordan graduated top of class from MIT but has no business experience.",
    "Taylor has 10 years in finance and raised a seed round quickly.",
    "Casey started a company out of high school and failed twice.",
]
y = ["successful", "failed", "successful", "failed"]
task = "Predict whether a startup founder will succeed."

tree = GPTree(
    task=task,
    generator="openai:gpt-5-mini",   # writes the questions
    answerer="jev",                  # answers them (Typesafe's Jev, the default)
)
tree.fit(X, y)
tree.predict(["Morgan is a former Google engineer with patents in machine learning."])
tree.predict_proba(X)
tree.plot()
tree.usage_.cost_usd
```

`fit` and `predict` work in scripts and notebooks alike; `afit`, `apredict` and `apredict_proba` are their async
twins. `X` is a list of texts or a DataFrame (each row becomes `column: value` lines).

## Models and providers

A model is `"provider:model"`, or an `LLM` with settings. A tuple is a fallback order.

```python
"openai:gpt-5-mini"                              # OPENAI_API_KEY
"anthropic:claude-sonnet-5-5"                    # ANTHROPIC_API_KEY, needs [anthropic]
"google:gemini-2.5-flash"                        # GEMINI_API_KEY
"xai:grok-4"                                     # XAI_API_KEY
"openrouter:meta-llama/llama-3.3-70b-instruct"   # OPENROUTER_API_KEY
"ollama:qwen3:8b"                                # local, no key
"jev"                                            # TYPESAFE_API_KEY
"openrouter:openai/gpt-6-luna-decisions"         # any decision model OpenRouter lists
```

Groq, Together, DeepSeek, Mistral, Azure and any OpenAI-compatible server (vLLM, LM Studio, llama.cpp) work too:

```python
from think_reason_learn import LLM

LLM("openai:gpt-5-mini", temperature=0.2, reasoning_effort="low", service_tier="flex")
LLM("Qwen3-32B", base_url="http://gpu-box:8000/v1")
```

Reasoning models write and answer; decision models (Jev, GPT-6 Luna Decisions, Microsoft-Decision-1...) only answer,
with probabilities. Putting a decision model in a writing role fails before anything is sent. Which models are
decision models is never guessed from their names: Typesafe serves only decision models, and OpenRouter's are read
from its public catalog, read once a day (a model it lists later is known the next day; until then, pass
`kind="decision"`). Elsewhere, ask for a model's decision API with
`LLM("openai:gpt-6-luna", kind="decision")`. Keys are read from the environment; a `base_url` of your own gets only
an `api_key` you pass. A reasoning model answers questions at temperature 0 unless you set one, so answers repeat, and
writes at its default.

## Speed, cost and reruns

```python
import think_reason_learn as trl

trl.set_limits("openai:gpt-5-mini", max_concurrency=64, rpm=5_000, tpm=4_000_000)
tree.estimate_cost(X, y).cost_usd  # what fit(X, y) would spend; no model is called
with trl.config_context(max_cost_usd=5):
    tree.fit(X, y)

tree.usage_.by("role", "model")    # calls, tokens and dollars
```

- Limits are per model and account, shared by every model in the process. Without limits, calls start at 8 in
  flight, halve on a rate limit and recover; `Retry-After` is honoured.
- Every answer is cached on disk, so nothing is paid for twice: reruns after a crash or a budget stop resume, and
  refitting with other settings reuses questions and answers. The first run says where the cache is;
  `trl.set_config(cache=False)` turns it off.
- `estimate_cost(X, y)` counts the calls, tokens and dollars of `fit(X, y)` (or of `predict(X)` without `y`)
  without calling a model, and without API keys. Each `fit` and `predict` logs its estimate first, as a warning
  above $1.
- `max_cost_usd` refuses work estimated over the cap, stops starting calls once spending passes it, and refuses a
  model with no known price, since it could not count it. Calls already sent finish, and one can cost more than
  estimated (a reasoning model thinking long), so the bill can pass the cap by what those calls cost. Models on your
  own machine or network are free.
- A failed call is retried up to 5 times, within 5 minutes of its first failure. When the last model left fails 5
  calls in a row, the run stops; a rerun continues from the cache. Missing answers are counted in a warning.

## scikit-learn

```python
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from think_reason_learn.rrf import RRF

# Two folds for the four samples above; use five on a real dataset.
f1 = make_scorer(f1_score, pos_label="successful")
cross_val_score(RRF(task=task, generator="openai:gpt-5-mini"), X, y, cv=2, scoring=f1)
```

Every model supports `clone`, `get_params`, `GridSearchCV`, `Pipeline`, pickling, and `save`/`load` (one readable
`model.json`, safe to load; API keys are never saved). `think_reason_learn.testing` fits and checks models offline with scripted fake models.

## Contributing

See [CONTRIBUTING.md](https://github.com/Vela-Research/think-reason-learn/blob/main/CONTRIBUTING.md).
Upgrading from 0.2? See the [migration guide](https://thinkreasonlearn.com/docs/getting_started/migration.html).

## License

MIT; see [LICENSE](https://github.com/Vela-Research/think-reason-learn/blob/main/LICENSE).
