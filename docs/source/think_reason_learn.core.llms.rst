Core LLM Interface
==================

Chat models (``OpenAIChoice``, ``GoogleChoice``, ``AnthropicChoice``, ``XAIChoice``) generate text through
``LLM.respond``. Jev (``JevChoice``), Typesafe's System One model, answers typed questions about a sample through
``LLM.answer`` and ``LLM.answer_many``: a ``NoulQuestion`` gets the probability of yes, a ``ChoiceQuestion`` one of
its labels. Jev needs ``TYPESAFE_API_KEY`` and is the default answering model in Random Rule Forest, GPTree and
Policy Induction.

Module contents
---------------

.. automodule:: think_reason_learn.core.llms
   :members:
   :show-inheritance:
   :undoc-members:

Type aliases
------------

.. autodata:: think_reason_learn.core.llms._schemas.LLMChoiceModel
   :annotation:

.. autodata:: think_reason_learn.core.llms._schemas.LLMChoiceDict
   :annotation:

.. autodata:: think_reason_learn.core.llms._schemas.LLMChoice
   :annotation:
