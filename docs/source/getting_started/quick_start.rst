Quick start
-----------

This fits a small GPTree on six example founder profiles, draws the tree and prints a prediction for each profile. Run
it in Jupyter, which allows ``await`` at the top level, with ``OPENAI_API_KEY`` set and Graphviz installed.

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
