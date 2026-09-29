About
-----

Think, Reason, Learn keeps the structure of decision trees, rule forests and short policies, and uses a language model
at each step to write or answer the questions. Every prediction therefore comes with the questions or rules that
produced it.

Asked directly, a language model gives an answer but no fixed model you can inspect. Here the fitted model is fixed and
can be read and checked. Unlike scikit-learn, its features are questions in plain language, asked of text.

The library is alpha software (version 0.1.0) from Vela Research, the research arm of Vela Partners, with the
University of Oxford.

Key features
~~~~~~~~~~~~

- **Readable reasoning**: every prediction comes with the questions, rules or policies behind it.
- **Asynchronous by design**: fitting and prediction run many language-model calls at once.
- **Choice of models**: OpenAI, Anthropic, Google and xAI models, set per step.

Core algorithms
~~~~~~~~~~~~~~~

- **GPTree**: decision trees whose questions a language model writes.
- **Random Rule Forest (RRF)**: an ensemble of yes or no questions written by a language model.
- **Policy Induction**: short policies learned from examples, which a model applies to new cases.
- **Reasoned Rule Mining**: plain-language if-then rules mined from a model's reasoning, combined into a calibrated, weighted ensemble.
- **Verifiable RL**: a learned policy decides which information to reveal next, or when to stop, and a classifier predicts from what it has seen.

The papers behind these methods are on the `research page <https://thinkreasonlearn.com/research>`_.
