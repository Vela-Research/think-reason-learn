Changelog
=========

Unreleased
----------
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

0.1.0
-----
- Initial public release of Think Reason Learn.


