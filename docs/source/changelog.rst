Changelog
=========

Unreleased
----------
The redesign: one way to call every model, built in steps. The new modules ship alongside the old ones until
every model uses them.

- ``think_reason_learn.llm``: one interface to every model. A model is a ``"provider:model"`` string
  (``"openai:gpt-5-mini"``, ``"anthropic:claude-sonnet-5-5"``, ``"ollama:qwen3:8b"``, ``"jev"``) or an
  ``LLM(...)`` with settings (temperature, reasoning effort, base URL, key, price). One OpenAI-compatible adapter
  covers OpenAI, Gemini, xAI, OpenRouter, Groq, Together, DeepSeek, Mistral, Azure, Ollama, vLLM, LM Studio and
  llama.cpp; Anthropic, Jev's System One format (also Microsoft-Decision-1 and Jev on OpenRouter) and OpenAI's
  Decisions API have their own. Decision models answer and cannot write: a role that writes rejects them.
- ``LLMSession`` makes every call: fallbacks, retries that honour ``Retry-After``, a shared limiter per model and
  account (``set_limits``; concurrency halves on a rate limit and recovers), a disk cache so no answer is paid
  for twice, one usage record per call with tokens and cost (``Usage``), and a run budget for every provider
  (``config_context(max_cost_usd=...)``). Calls run on one event loop owned by the library, so sync code works in
  notebooks too.
- ``think_reason_learn.testing``: ``FakeLLM`` and ``fake_llms`` fit and check models offline, with scripted
  replies and injected failures.
- ``TRLClassifier`` (and ``BinaryTRLClassifier`` for two-class models): the scikit-learn base of every model; the
  old ``TRLModel`` is removed. ``fit``, ``predict`` and ``predict_proba`` work in
  scripts and notebooks, with ``afit``, ``apredict`` and ``apredict_proba`` for async code; settings are stored
  untouched so ``clone``, ``get_params`` and grid search work; any labels; ``usage_`` records every call;
  ``save``/``load`` write ``model.joblib`` plus a readable ``model.json`` and never store API keys.
  ``think_reason_learn.testing.check_classifier`` checks a model against this contract offline.
- **GPTree, rebuilt** on ``TRLClassifier``: ``GPTree(task=..., generator=..., answerer="jev")`` with ``fit``,
  ``predict``, ``predict_proba``, ``apply``, ``plot``, ``export_graphviz``, ``prune`` and ``write_instructions``.
  Closer to the paper: insights per class before growing, defaults ``max_depth=18`` and ``min_samples_leaf=0.005``
  (a fraction of the samples, as in scikit-learn), yes/no splits unless ``max_choices`` allows more. Fixes: earlier
  answers no longer leak into prompts, a sample whose answer fails follows the largest branch instead of
  vanishing, an empty branch no longer disqualifies a question, and a split must decrease impurity. Faster: the
  nodes of a level grow at once and answers are reused across nodes and between fit and predict. New
  ``min_samples_split``, ``min_impurity_decrease``, ``max_leaf_nodes`` and ``criterion="entropy"``; the critic is
  ``critic=True`` or a model. Removed: ``qgen_instr_llmc`` (unused), ``fit_iter``, ``resume_fit``, ``stop``,
  ``decision_threshold`` (use scikit-learn's ``TunedThresholdClassifierCV``) and the code question type.
- **RRF, rebuilt** on ``TRLClassifier``, following the paper by default: questions are written from a held-out,
  class-balanced generation set (``generation_size=500``; 25 prompts of 20 examples and 10 questions), so they are
  never scored on samples the generator saw; questions are ranked by F0.5, those at or below the positive rate and
  near-duplicates by Hamming distance (``hamming_threshold=0.15``) are dropped; and the default ``combiner="vote"``
  predicts positive when at least T of the top N questions say YES, with (N, T) tuned by cross-validation.
  ``combiner="elasticnet"`` remains. Prediction answers only the questions used. Fixes: large datasets no longer
  get zero questions, ranking and tuning use one ``beta``, and fit and predict ask questions the same way.
  ``screening_size`` keeps the cost-saving screening step; ``answers(X)`` shows each rule's answers. Removed:
  ``predict_founder_level`` (use ``predict``/``predict_proba``), ``cross_validate_aggregation`` (refitting reuses
  cached questions and answers, and scikit-learn's ``cross_val_score`` works), prompt presets, cumulative memory and
  embedding-based filtering.
- **Policy Induction, rebuilt** on the new base, following the paper by default: a first policy written from 40
  balanced examples (or ``initial_policy`` from an expert), refined with 240 balanced cases over 4 rounds, and a
  refinement kept only when it raises precision on those cases (the paper's memory step, missing before). The
  answerer judges each sample under the whole policy; ``predictor="rules"`` keeps the per-rule L1 logistic
  regression. ``batch_size`` trades fidelity for fewer calls. Removed: ``confirm_requests`` (it called
  ``input()``), checkpoints (the cache resumes), ``WeightTrainerConfig`` (plain parameters).
- **Reasoned Rule Mining, rebuilt** on the new base. It mines one IF-THEN rule per labelled sample (reasoning and
  rule in one call instead of two), compiles them into a policy, and applies the paper's cascade: the answerer
  judges each sample under the policy, and samples it rates at or above ``harsh_gate=0.95`` get a harsh second
  look, by ``harsh_answerer`` if given (the paper's cheap-then-strong cascade, which could not be configured before).
  Probabilities come from decision models or an ensemble's vote share instead of token log probabilities, so every
  provider works (OpenAI and Anthropic could not fit before). One logistic regression, fitted on samples the miner
  never saw, calibrates and weighs both stages, replacing Platt scaling plus a weight grid that was 95% redundant.
  Removed: ``RRMConfig`` (plain parameters), the perplexity filter and the rolling memory.
- ``core._combine``: one logistic combiner and F-beta threshold search shared by the models. ``C`` is chosen by
  log loss (average precision ties on separable data and picked an over-regularised model), and thresholds are tuned
  on out-of-fold scores of the final model's settings, midway between scores so they are not knife-edged.
- Requires ``openai`` 3.x and ``anthropic`` 1.x. The Anthropic SDK no longer takes ``temperature``; the old
  adapter now sends it in the request body when set.

0.2.0
-----
**Breaking:** code that relied on answers defaulting to the generation model now answers with Jev and needs
``TYPESAFE_API_KEY`` (or pass the chat model as ``qanswer_llmc`` / ``predict_llmc``), and new Random Rule Forest
instances use the elastic-net combiner unless ``aggregation_method="vote"`` is passed.

- Jev, Typesafe's System One model, is the default answering model: ``qanswer_llmc`` in Random Rule Forest and
  GPTree and ``predict_llmc`` in Policy Induction default to ``[JevChoice()]`` instead of the generation model.
  Question and policy generation, critics and instructions still use chat models.
- Jev answers need ``TYPESAFE_API_KEY``. Without it, constructing a method that answers with Jev raises
  ``MissingAPIKeyError`` with a message showing how to answer with a chat model instead. Saved models keep the
  answering model they were saved with, and load without the key so they can be inspected.
- Jev sends each sample once with all its questions. Random Rule Forest records Jev's probability of YES as YES at or
  above 0.5 and keeps the probabilities (``get_answer_probabilities()``); Policy Induction records its policy scores
  as YES/NO at 0.5 the same way, without keeping the probabilities; GPTree asks Jev multiple-choice questions whose
  labels are the question's choices.
- Jev must come first in the answering models when it is used; chat models may follow it as fallbacks. A Jev run in
  which every request fails raises ``LLMError`` instead of fitting on no answers.
- Random Rule Forest keeps answers aligned with their samples when ``X`` has an index other than ``0..n-1``; the
  answers tables are indexed by position.
- Every ``fit()`` and ``predict()`` prints its estimated Jev cost before spending (250 tokens per request plus request
  bytes / 4.2, with each non-ASCII character counted as a token; calibrated on billed usage) and stops if the estimate or the actual spend passes
  ``JevChoice(max_cost_usd=10)``. Paid answers are cached in ``~/.cache/think_reason_learn/jev``;
  ``JevChoice(cache=False)`` turns the cache off.
- Random Rule Forest's default combiner is now elastic-net (``aggregation_method="elasticnet"``) instead of the
  top-K / T vote, so by default the founder-level model is learned from Jev's probabilities.
  ``aggregation_method="vote"`` keeps the vote; saved models load with their saved method, or ``"vote"`` if none was
  saved. ``predict_founder_level()`` on an elastic-net model returns ``prediction``, ``probability`` and
  ``threshold``; ``yes_count``, ``k`` and ``t`` belong to the vote, and passing ``k`` or ``t`` to an elastic-net model
  raises. Its decision threshold is now chosen among the fitted probabilities, so rare positives are not all predicted
  NO, and inner CV uses no more folds than the smaller class has samples. Prediction asks only the questions the
  fitted model weighs, so questions added, filtered or excluded after ``fit()`` change nothing until the model is
  refitted; a warning says so.
- Elastic-net's default ``elasticnet_cs`` is now ``(1.0,)`` instead of ``(0.05, 0.1, 0.5)``. On twelve benchmark
  datasets a fixed C=1 gave a mean test AUC of 0.687 with Jev's probabilities against 0.673 (better on 10 of 12) and
  0.665 against 0.663 with YES/NO answers; the inner CV over the old grid chose too much regularisation on small
  training sets and sometimes zeroed every weight. Those datasets had 14 to 16 questions each; with many questions
  relative to training rows, or a large training set, pass a grid, e.g. ``elasticnet_cs=(0.01, 0.1, 1.0, 10.0)``.
- Random Rule Forest's elastic-net combiner (``aggregation_method="elasticnet"``) uses Jev's probability of YES as
  each question's feature (``answer_features="probability"``, the default); ``answer_features="binary"`` keeps
  YES/NO at 0.5. The vote combiner still counts YES answers, question metrics and similarity filters still use
  YES/NO, ``predict()`` still yields YES/NO, and models saved before this setting load with ``"binary"``.
- New ``LLM.answer`` and ``LLM.answer_many`` answer typed questions (``NoulQuestion``, ``ChoiceQuestion``) with Jev,
  falling back to chat models.
- ``httpx`` is now a direct dependency.
- ``cross_validate_aggregation(method="elasticnet")`` cross-validates the elastic-net combiner.
- Elastic-net (Random Rule Forest, ``cross_validate_aggregation``) and Policy Induction's weight fit pick their
  scikit-learn arguments by version, so they keep working on scikit-learn 1.8 and later.
- Policy Induction answering with a chat model judges a sample against up to ``policy_batch_size`` policies (default
  10) in one call, for scoring and ``predict()`` alike; ``policy_batch_size=1`` keeps one call per policy. Answers
  missing from a batched reply are asked again one policy at a time. Models saved before this load with
  ``policy_batch_size=1``, the way they were fitted.
- Policy Induction's new ``max_gen_batches`` caps the generation rounds; the default ``None`` runs until a class
  runs out, as before.
- Policy Induction's ``predict()`` skips a sample for which no policy answer could be obtained, instead of predicting
  from an all-NO vector, and keeps its checkpoint so a later run retries it. A checkpoint is resumed only for the same
  samples. Scoring warns how many (policy, sample) cells got no answer; they count as NO.
- ``save()`` on Random Rule Forest and GPTree writes to ``save_path`` itself, as Policy Induction already did,
  instead of ``<save_path>/<name>``. By default that is ``./rrfs/<name>`` or ``./gptrees/<name>`` (previously one level
  deeper), and a loaded model saves back to the directory it was loaded from instead of nesting a new directory
  inside it on every load and save. Pass ``dir_path`` to ``save()`` to choose another directory. A loaded Reasoned Rule
  Mining model also saves back to the directory it was loaded from instead of its parent.
- Policy Induction accepts ``yes`` / ``no`` labels in any case, as Random Rule Forest does.
- **Breaking:** Random Rule Forest, GPTree, Policy Induction and Reasoned Rule Mining subclass the new abstract base
  class ``think_reason_learn.core.TRLModel``, which requires ``set_task``, ``fit``, ``predict``, ``save`` and ``load``.
  For one API across the four:

  - ``GPTree.fit()`` builds the whole tree and returns it instead of yielding nodes. Replace
    ``async for node in tree.fit(X, y)`` with ``await tree.fit(X, y)``, or with
    ``async for node in tree.fit_iter(X, y)`` to keep getting each node as it is built, for example to ``stop()`` the
    tree or give it ``advice()`` between nodes. ``resume_fit()`` still yields nodes.
  - Random Rule Forest and GPTree take their task with ``set_task(task_description, instructions_template=None)``, as
    Policy Induction does. ``set_tasks(instructions_template, task_description)``, with the arguments in the old
    order, still works but is deprecated and warns.
  - ``GPTree.load()`` takes ``dir_path`` instead of ``path``. Positional calls are unchanged; replace
    ``GPTree.load(path=...)`` with ``GPTree.load(dir_path=...)``.
- A chat model's empty or refused reply (no structured answer, blank text, or text where a structured answer was
  asked for) now falls back to the next model in the priority list instead of being returned as the answer. If every
  model replies that way, the response has ``response=None``, as callers already expect. The returned reply's
  ``total_tokens`` includes the replies discarded before it, so token usage still counts them. Anthropic replies are read
  from the tool call or text blocks wherever they are in the message, and an Anthropic reply with neither raises an
  ``LLMError`` naming its stop reason instead of an ``IndexError``.
- GPTree's question critic now works: with ``use_critic=True``, ``critic_llmc`` reviews each node's candidate questions
  (one extra call per node, seeing the same samples as generation) and may drop, merge or reword questions that can't
  be answered from a sample's text, leak the label, are ambiguous or compound, or duplicate another, and fix choices
  that overlap or miss cases. It never adds questions; if it fails, or returns none or more than it was given, the
  generated questions are kept with a warning. ``use_critic`` still defaults to ``False``, which changes nothing.
- GPTree rejects Jev in ``qgen_llmc``, ``qgen_instr_llmc`` or ``critic_llmc`` when the tree is created; Jev only
  answers questions.
- GPTree's ``critic_llmc`` and ``qgen_instr_llmc`` are optional and default to ``qgen_llmc``, so
  ``GPTree(qgen_llmc=[...])`` is enough to build a tree.
- Samples with the same text and questions are sent to Jev and paid for once when they are answered together (e.g.
  rows repeated in one ``predict()``), and each copy gets the answers. The cost estimate and cap count them once, and
  the cost line says how many duplicates were not sent.
- A run whose Jev answers come from two releases (with ``model="jev-latest"``, cached answers keep the release that
  gave them) logs one warning naming both, and how to answer with one release.
- ``predict_founder_level()`` adds an ``n_answered`` column with how many of the questions asked for each sample
  were answered. Unanswered questions (e.g. a failed Jev request) count as NO, and a warning says how many samples
  have them.
- **Behaviour change:** ``cross_validate_aggregation`` now evaluates Random Rule Forest's default model by default:
  ``method="elasticnet"`` instead of ``"vote"``, and ``elasticnet_cs=(1.0,)`` instead of ``(0.05, 0.1, 0.5)``, with
  ``method``, ``metric`` (now ``"f1"`` instead of ``"f_beta"``), ``elasticnet_cs``, ``elasticnet_l1_ratios`` and
  ``elasticnet_cv`` defaulting to Random Rule Forest's own. Pass ``method="vote"`` for the top-K / T vote, or
  ``metric="f_beta", beta=0.5`` (or ``"recall"``, ``"precision"``, ...) to tune for another metric. Its elastic-net fit is now Random Rule Forest's (the threshold
  is chosen among the fitted probabilities instead of a 0.05 to 0.95 grid), so each fold's model is the one
  ``RRF(aggregation_method="elasticnet")`` fits on that fold's training split.
- ``cross_validate_aggregation`` accepts Jev's probabilities (``RRF.get_answer_probabilities()``) as well as YES/NO
  answers. Elastic-net uses the probabilities as features, as ``answer_features="probability"`` does; the vote counts a
  probability at or above 0.5 as YES. Values that are neither, a mix of the two, or labels other than one ``"YES"`` or
  ``"NO"`` (any case) per row with both present raise ``ValueError``; missing answers still count as NO.
- ``cross_validate_aggregation(method="elasticnet")`` skips, with a warning, a fold whose training split holds one
  class instead of failing in scikit-learn, and fits without inner CV when the smaller class has one training sample.
- A temperature of ``0`` now reaches OpenAI, Anthropic, Google and xAI models. It was treated as "not given", so
  those calls ran at the provider's default temperature (usually 1) instead. A model that rejects a temperature, such
  as OpenAI's reasoning models, is retried once without it and then uses its default for the rest of the process,
  with a warning.
- Reasoned Rule Mining reads token log probabilities, which OpenAI and Anthropic models do not return here. With
  one of them as ``extract_llmc`` or ``predict_llmc``, ``fit`` dropped every rule, scored every vote 0.5 and then
  failed. ``fit`` and ``predict`` now raise ``ValueError`` before any call, and a model that returns no log
  probabilities raises ``LLMError`` instead of being scored 0.5.

0.1.0
-----
- Initial public release of Think Reason Learn.


