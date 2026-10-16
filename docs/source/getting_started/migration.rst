Migrating from 0.2
------------------

0.3 is a redesign: one way to call every model, on one provider layer, as scikit-learn classifiers. Most changes are
renames; this page lists them all.

Every model
~~~~~~~~~~~

======================================================  =====================================================
0.2                                                     0.3
======================================================  =====================================================
``OpenAIChoice(model="gpt-4.1-nano")`` and the like      ``"openai:gpt-4.1-nano"``, or ``LLM(...)`` with settings
a list of choices (fallbacks)                            a tuple of models
``*_llmc`` that write (``qgen_llmc``, ``gen_llmc``,      ``generator=``
``reason_llmc``)
``qanswer_llmc``, ``predict_llmc``                       ``answerer=`` (default ``"jev"``)
``*_temperature``                                        ``LLM(model, temperature=...)``
``llm_semaphore_limit``                                  ``set_limits(model, max_concurrency=, rpm=, tpm=)``
``await model.set_task(task_description)``               ``task=`` in the constructor; ``write_instructions()``
                                                         to read the instructions before fitting
``await model.fit(X, y)``                                ``model.fit(X, y)``, or ``await model.afit(X, y)``
``async for ... in model.predict(X)``                    ``predict(X)``, ``predict_proba(X)``, ``apredict(X)``
labels ``"YES"`` / ``"NO"`` only                         any labels; ``pos_label=`` for two-class models
``token_usage``                                          ``usage_`` (tokens and cost per call)
``save_path=``, ``name=``, ``save(dir_path)``            ``save(path)``, ``Model.load(path)``
``JevChoice(max_cost_usd=...)``                          ``config_context(max_cost_usd=...)`` (any provider)
``.env`` read from the working directory                 environment variables only
``think_reason_learn.core.llms``                         ``think_reason_learn.llm``
``think_reason_learn.core.exceptions``                   ``think_reason_learn.exceptions``
``TRLModel``                                             ``TRLClassifier``
======================================================  =====================================================

GPTree
~~~~~~

====================================================  =======================================================
0.2                                                   0.3
====================================================  =======================================================
``critic_llmc`` and ``use_critic``                    ``critic=True`` (the generator) or a model
``qgen_instr_llmc``                                   removed (never used)
``n_samples_as_context``                              ``n_examples``
``max_node_width``                                    ``max_choices``
``decision_threshold``                                removed; use ``sklearn.model_selection.TunedThresholdClassifierCV``
``fit_iter``, ``resume_fit``, ``stop``                removed; reruns resume from the answer cache
``view_node``, ``get_questions()``, ``get_node()``    ``plot()``, ``export_graphviz()``, ``tree_``
``prune_tree(node_id)``                               ``prune(node_id)``
defaults: depth unlimited, 1 sample per leaf          the paper's: ``max_depth=18``, ``min_samples_leaf=0.005``
====================================================  =======================================================

Random Rule Forest
~~~~~~~~~~~~~~~~~~

=====================================================  =====================================================
0.2                                                    0.3
=====================================================  =====================================================
``max_generated_questions``                            ``n_prompts`` × ``questions_per_prompt``
``max_samples_as_context``                             ``n_examples``
``aggregation_method`` (default ``"elasticnet"``)      ``combiner`` (default ``"vote"``, as in the paper)
``question_scoring_f_beta``, ``aggregation_metric``    ``beta`` (default 0.5)
``predict_founder_level``                              ``predict``, ``predict_proba``; ``answers(X)``
``cost_sensitive`` and its config                      ``screening_size``, ``screened_questions``
``cross_validate_aggregation``                         refit with other settings (answers are cached) or
                                                       ``sklearn.model_selection.cross_val_score``
prompt presets, cumulative memory, semantic filter     removed
=====================================================  =====================================================

Policy Induction
~~~~~~~~~~~~~~~~

=====================================================  =====================================================
0.2                                                    0.3
=====================================================  =====================================================
``config=WeightTrainerConfig(...)``                    ``predictor="rules"`` with ``beta`` and ``cv``
``max_policy_length``                                  ``max_rules``
``max_gen_batches``, ``policy_batch_size``             ``n_rounds``, ``n_samples``, ``batch_size``
``confirm_requests``                                   removed; set ``max_cost_usd``
=====================================================  =====================================================

Reasoned Rule Mining and VerifiableRL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``RRMConfig`` and ``VerifiableRLConfig`` become constructor parameters.
- Reasoned Rule Mining no longer reads token log probabilities; ``harsh_answerer`` picks the second-look model.
- VerifiableRL takes a 2-D array (``slots`` gives each slot's columns) and needs ``pip install
  'think-reason-learn[rl]'``; ``predict_proba`` has two columns; ``pretrain_classifier`` is part of ``fit``
  (``pretrain_epochs``).
