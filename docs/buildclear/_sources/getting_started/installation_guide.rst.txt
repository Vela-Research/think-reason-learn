Installation Guide
==================

Prerequisites
-------------

- Python 3.13 or higher
- pip (latest version recommended)

Standard Installation
---------------------

.. code-block:: bash

   pip install think-reason-learn

From Source
-----------

.. code-block:: bash

   git clone https://github.com/vela-research/think-reason-learn.git
   cd think-reason-learn
   poetry install

Development Setup
-----------------

For contributing or running tests/docs:

.. code-block:: bash

   poetry install --with dev,docs
   poetry run pre-commit install  # Optional: code quality hooks

Troubleshooting
---------------

- If you encounter dependency issues, ensure your Python version matches.
- For LLM integrations, set API keys as environment variables (OPENAI_API_KEY, GOOGLE_AI_API_KEY, XAI_API_KEY, ANTHROPIC_API_KEY).
- See :doc:`/contributing` for more dev tips.
