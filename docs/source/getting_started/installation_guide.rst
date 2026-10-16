Installation guide
------------------

Think, Reason, Learn needs Python 3.13 or later.

.. code-block:: bash

   pip install think-reason-learn              # OpenAI-compatible providers and Jev
   pip install 'think-reason-learn[anthropic]'  # + Claude
   pip install 'think-reason-learn[rl]'         # + PyTorch, for VerifiableRL
   pip install 'think-reason-learn[all]'

Drawing trees with ``GPTree.plot`` needs `Graphviz <https://graphviz.org/download/>`_
(``brew install graphviz`` or ``apt-get install graphviz``).

API keys
~~~~~~~~

Keys are read from the environment, by provider:

================  ===================================================
Provider          Variable
================  ===================================================
OpenAI            ``OPENAI_API_KEY``
Anthropic         ``ANTHROPIC_API_KEY``
Google Gemini     ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)
xAI               ``XAI_API_KEY``
OpenRouter        ``OPENROUTER_API_KEY``
Groq              ``GROQ_API_KEY``
Together          ``TOGETHER_API_KEY``
DeepSeek          ``DEEPSEEK_API_KEY``
Mistral           ``MISTRAL_API_KEY``
Azure             ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_BASE_URL``
Typesafe (Jev)    ``TYPESAFE_API_KEY``
Ollama            none (``OLLAMA_BASE_URL`` for a remote server)
================  ===================================================

Pass ``LLM(..., api_key=...)`` to use another key. The library does not read ``.env`` files; load one in your
application (for example with ``python-dotenv``) if you keep keys there.

Jev, Typesafe's decision model, answers questions by default (``answerer="jev"``). See
`docs.typesafe.ai <https://docs.typesafe.ai>`_ for access and prices, or answer with any reasoning model instead.

From source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/Vela-Research/think-reason-learn.git
   cd think-reason-learn
   poetry install --with dev,docs --all-extras
   poetry run pytest
