Quick start
-----------

Fit a GPTree on a handful of founder profiles. The generator writes questions, the answerer answers them for each
founder, and the tree learns which answers predict success.

.. code-block:: python

   from think_reason_learn.gptree import GPTree

   X = [
       "Alex is a serial entrepreneur with two exits and a strong network in AI.",
       "Jordan graduated top of class from MIT but has no business experience.",
       "Taylor has 10 years in finance and raised a seed round quickly.",
       "Casey started a company out of high school and failed twice.",
       "Morgan is a former Google engineer with patents in machine learning.",
       "Riley left a consulting job last month and has no product yet.",
   ]
   y = ["successful", "failed", "successful", "failed", "successful", "failed"]

   tree = GPTree(
       task="Predict whether a startup founder will succeed.",
       generator="openai:gpt-5-mini",
       answerer="jev",
       min_samples_leaf=1,
   )
   tree.write_instructions()      # optional: read or edit before the expensive fit
   tree.fit(X, y)

   tree.predict(["Sam is a staff engineer who sold a company."])
   tree.predict_proba(X)
   tree.plot()                    # needs Graphviz
   tree.usage_.by("role", "model")

The other models are called the same way:

.. code-block:: python

   from think_reason_learn.policy_induction import PolicyInduction
   from think_reason_learn.reasoned_rule_mining import ReasonedRuleMining
   from think_reason_learn.rrf import RRF

   RRF(task=task, generator="openai:gpt-5-mini").fit(X, y)
   PolicyInduction(task=task, generator="anthropic:claude-sonnet-5-5").fit(X, y)
   ReasonedRuleMining(task=task, generator="openai:gpt-5-mini", harsh_answerer="openai:gpt-5").fit(X, y)

Choosing models
~~~~~~~~~~~~~~~

A model is ``"provider:model"`` or an ``LLM`` with settings; a tuple is a fallback order. Reasoning models write and
answer; decision models such as Jev only answer, with probabilities.

.. code-block:: python

   from think_reason_learn import LLM

   GPTree(
       task=task,
       generator=("anthropic:claude-sonnet-5-5", "openai:gpt-5-mini"),
       answerer=LLM("ollama:qwen3:8b", questions_per_call=5),
       critic=True,
   )

Async code
~~~~~~~~~~

``fit`` and ``predict`` work in notebooks and scripts. In async applications use ``afit``, ``apredict`` and
``apredict_proba``.

Offline tests
~~~~~~~~~~~~~

.. code-block:: python

   from think_reason_learn.testing import FakeLLM, fake_llms

   writer = FakeLLM(generate=lambda request: {"questions": [{"text": "Is the founder technical?"}]})
   judge = FakeLLM(answer=lambda sample, question: 0.9 if "engineer" in sample else 0.1, kind="decision")
   with fake_llms(writer=writer, judge=judge):
       GPTree(task=task, generator="fake:writer", answerer="fake:judge").fit(X, y)
