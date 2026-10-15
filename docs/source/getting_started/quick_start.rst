Quick start
-----------

This fits a small GPTree on six example founder profiles, draws the tree and prints a prediction for each profile. Run
it in Jupyter, which allows ``await`` at the top level, with ``OPENAI_API_KEY`` and ``TYPESAFE_API_KEY`` set and
Graphviz installed.

.. code-block:: python

   import pandas as pd
   from IPython.display import Image, display
   from think_reason_learn.gptree import GPTree
   from think_reason_learn.core.llms import OpenAIChoice

   X = pd.DataFrame({"founder_info": [
       "Alex is a serial entrepreneur with two successful exits and expertise in AI.",
       "Jordan graduated top of class from Oxford but has no business experience.",
       "Taylor has 10 years in finance, raised a seed round quickly and built a strong team.",
       "Casey started a company out of high school and has faced several failures.",
       "Morgan is a former Google engineer with machine learning patents and VC backing.",
       "Seraphine has a first-class degree in Minerals Engineering and a strong mining network.",
   ]})
   y = ["successful", "failed", "successful", "failed", "successful", "successful"]

   llm = [OpenAIChoice(model="gpt-4o-mini")]
   tree = GPTree(qgen_llmc=llm, critic_llmc=llm, qgen_instr_llmc=llm, max_depth=2)
   await tree.set_tasks(task_description="Predict whether a founder succeeds or fails from their background.")

   async for node in tree.fit(X, y, reset=True):
       pass  # each step yields the node just built

   display(Image(tree.view_node(tree.get_root_id())))
   print(tree.get_questions())

   async for index, question, answer, node_id, usage in tree.predict(X):
       print(index, question, answer)

Other providers work the same way. Use ``AnthropicChoice``, ``GoogleChoice`` or ``XAIChoice`` from
``think_reason_learn.core.llms`` with the matching API key. The API reference covers
:doc:`Random Rule Forest </think_reason_learn.rrf>` and the other methods.

Who answers the questions
~~~~~~~~~~~~~~~~~~~~~~~~~

The chat model above writes the tree's questions. Jev, Typesafe's System One model, answers them for each profile:
it is the default answering model in GPTree, Random Rule Forest (``qanswer_llmc``) and Policy Induction
(``predict_llmc``). Jev reads one sample and answers all the questions about it in one request, so the sample is paid
for once rather than once per question. Labels are never sent to it.

Before a run spends anything, the library prints the estimated cost. Each ``fit()`` or ``predict()`` stops if the
estimate, or the actual spend, passes its cap, which is $10 unless you set another. Answers already paid for are kept
in ``~/.cache/think_reason_learn/jev`` and reused when the same sample and questions come up again:

.. code-block:: python

   from think_reason_learn.core.llms import JevChoice

   tree = GPTree(qgen_llmc=llm, critic_llmc=llm, qgen_instr_llmc=llm,
                 qanswer_llmc=[JevChoice(max_cost_usd=25, cache=False)])

To answer with a chat model instead, for example because you have no ``TYPESAFE_API_KEY``, pass it as the answering
model:

.. code-block:: python

   tree = GPTree(qgen_llmc=llm, critic_llmc=llm, qgen_instr_llmc=llm, qanswer_llmc=llm, max_depth=2)

Why Jev: on twelve benchmark datasets we had studied before, with the same Random Rule Forest questions and the same
logistic combiner, Jev's probabilities gave a mean test AUC of 0.686, against 0.660 for Jev's answers recorded as
YES/NO at 0.5 and 0.630 for Gemini's YES/NO answers. Random Rule Forest's default combiner, elastic-net, uses the
probabilities. For Policy Induction, Jev's probabilities gave 0.695 against Gemini's 0.683. This is a development
comparison, not an independent evaluation. As for cost, answering 2,194 court cases (ECHR) with 14 questions each
through this library used 3.16 million input tokens: $0.13 at Jev's list price of $42 per billion input tokens
(checked 22 September 2026).
