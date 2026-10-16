Contributing
============

Thank you for helping. Fork the repository, work on a branch, and open a pull request against ``main``.

Set up
------

.. code-block:: bash

   git clone https://github.com/<you>/think-reason-learn.git
   cd think-reason-learn
   poetry install --with dev,docs --all-extras
   poetry run pre-commit install


Before you open a pull request
------------------------------

.. code-block:: bash

   poetry run ruff check . && poetry run ruff format --check .
   poetry run pyright
   poetry run pytest          # offline: no test calls a real provider


Add a line to ``docs/source/changelog.rst`` for anything a user would notice.

How the code is laid out
------------------------

- ``think_reason_learn/llm``: one interface to every model. A provider adapter turns a request into one call; the
  session around it does fallbacks, retries, limits, the cache, usage and budgets for all of them.
- ``think_reason_learn/core``: ``TRLClassifier``, the scikit-learn base every model builds on, and the helpers models
  share (rendering samples, labels, example batches, prompts, logistic combiners and thresholds).
- One package per model (``gptree``, ``rrf``, ``policy_induction``, ``reasoned_rule_mining``, ``verifiable_rl``).
- ``tests/`` mirrors the packages.

Adding a provider
-----------------

- If it speaks OpenAI's Chat Completions, add one entry to ``_PROVIDERS`` in ``llm/_spec.py``.
- Otherwise, write an adapter in ``llm/_adapters/`` with ``generate`` and/or ``answer``, map its errors with
  ``classify``, and register it in ``ADAPTERS``. Test it on the wire with the ``wire`` fixture (see
  ``tests/llm/test_adapters.py``).

Adding a model
--------------

Subclass ``TRLClassifier`` (or ``BinaryTRLClassifier``), store every constructor argument unchanged, and implement
``_afit`` and ``_apredict_proba``. Make every model call through the ``LLMSession`` you are given. Test it offline with
``think_reason_learn.testing``: ``fake_llms`` for scripted models, and ``check_classifier`` for the shared contract.

Code of conduct
---------------

Be respectful and inclusive. Report unacceptable behaviour to the maintainers.
