Installation Guide
------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.13 or higher
- pip (latest version recommended)
- Graphviz, to draw trees (``brew install graphviz`` or ``apt-get install graphviz``)

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install think-reason-learn

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/vela-research/think-reason-learn.git
   cd think-reason-learn
   poetry install

Development Setup
~~~~~~~~~~~~~~~~~

For contributing or running tests/docs:

.. code-block:: bash

   poetry install --with dev,docs
   poetry run pre-commit install  # Optional: code quality hooks

API keys
~~~~~~~~

Two kinds of model do the work:

- A chat model writes the questions, policies and instructions. Set the key for the provider you choose:
  ``OPENAI_API_KEY``, ``GOOGLE_AI_API_KEY``, ``ANTHROPIC_API_KEY`` or ``XAI_API_KEY``.
- Jev answers those questions about each sample, by default. Jev is Typesafe's System One model, and its key is
  ``TYPESAFE_API_KEY``; see `docs.typesafe.ai <https://docs.typesafe.ai>`_ for access and prices. Without the key,
  creating an RRF, GPTree or PolicyInduction raises an error that shows how to answer with a chat model instead.

Set the keys in your environment or in a ``.env`` file in the folder you run from, and keep that file out of version
control:

.. code-block:: bash

   # .env
   OPENAI_API_KEY=...
   TYPESAFE_API_KEY=...

Troubleshooting
~~~~~~~~~~~~~~~

- If you encounter dependency issues, ensure your Python version matches.
- For LLM integrations, set API keys as environment variables (OPENAI_API_KEY, GOOGLE_AI_API_KEY, XAI_API_KEY,
  ANTHROPIC_API_KEY, and TYPESAFE_API_KEY for Jev).
- See :doc:`/contributing` for more dev tips.
