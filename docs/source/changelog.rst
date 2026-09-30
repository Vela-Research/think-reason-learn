Changelog
=========

Unreleased
----------
- Jev, Typesafe's System One model, is the default answering model: ``qanswer_llmc`` in Random Rule Forest and
  GPTree and ``predict_llmc`` in Policy Induction default to ``[JevChoice()]`` instead of the generation model.
  Question and policy generation, critics and instructions still use chat models.
- Jev answers need ``TYPESAFE_API_KEY``. Without it, constructing a method that answers with Jev raises
  ``MissingAPIKeyError`` with a message showing how to answer with a chat model instead. Saved models keep the
  answering model they were saved with.
- Jev sends each sample once with all its questions. Random Rule Forest records Jev's probability of YES as YES at or
  above 0.5 and keeps the probabilities (``get_answer_probabilities()``); Policy Induction records its policy scores
  as YES/NO at 0.5 the same way, without keeping the probabilities; GPTree asks Jev multiple-choice questions whose
  labels are the question's choices.
- Jev must come first in the answering models when it is used; chat models may follow it as fallbacks. A Jev run in
  which every request fails raises ``LLMError`` instead of fitting on no answers.
- Random Rule Forest keeps answers aligned with their samples when ``X`` has an index other than ``0..n-1``; the
  answers tables are indexed by position.
- Every ``fit()`` and ``predict()`` prints its estimated Jev cost before spending and stops if the estimate or the
  actual spend passes ``JevChoice(max_cost_usd=10)``. Paid answers are cached in ``~/.cache/think_reason_learn/jev``;
  ``JevChoice(cache=False)`` turns the cache off.
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


