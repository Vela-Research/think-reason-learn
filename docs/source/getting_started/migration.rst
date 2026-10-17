Migrating from 0.2
------------------

0.3 is a redesign: one way to call every model, on one provider layer, as scikit-learn classifiers. Most changes are
renames; this page lists them. The defaults are 0.2's, except where a row says otherwise.

Every model
~~~~~~~~~~~

==============================================================  =======================================================================
0.2                                                             0.3
==============================================================  =======================================================================
``OpenAIChoice(model="gpt-4.1-nano")`` and the like             ``"openai:gpt-4.1-nano"``, or ``LLM(...)`` with settings
a list of choices (fallbacks)                                   a tuple of models
``qgen_llmc``, ``gen_llmc``, ``reason_llmc``, ``extract_llmc``  ``generator=``
``qanswer_llmc``, ``predict_llmc``                              ``answerer=`` (default ``"jev"``, which needs ``TYPESAFE_API_KEY``)
``*_temperature``                                               ``LLM(model, temperature=...)``
``qanswer_batch_size``, ``policy_batch_size``                   ``LLM(model, questions_per_call=...)``
``llm_semaphore_limit``                                         ``set_limits(model, max_concurrency=, rpm=, tpm=)``
``await model.set_task(...)``, ``set_tasks``                    ``task=`` in the constructor
``*_instructions_template``                                     ``write_instructions()`` to read them before fitting; ``instructions=``
``await model.fit(X, y)``                                       ``model.fit(X, y)``, or ``await model.afit(X, y)``
``async for ... in model.predict(X)``                           ``predict(X)``, ``predict_proba(X)``, ``apredict(X)``
labels ``"YES"`` / ``"NO"`` only                                any labels; ``pos_label=`` for two-class models
``max_samples_as_context``, ``n_samples_as_context``            ``n_examples``
``class_ratio``                                                 ``example_class_ratio`` (GPTree, RRF); Policy Induction always balances
``*_update_interval`` (progress logs)                           ``verbose=1`` (a progress bar)
``token_usage``                                                 ``usage_`` (tokens and cost per call)
``save_path=``, ``name=``, ``save(dir_path)``                   ``save(path)``, ``Model.load(path)``; 0.2 saves do not load (below)
``JevChoice(max_cost_usd=...)``, $10 by default                 ``config_context(max_cost_usd=...)``, no cap unless set
``JevChoice(cache=False)``                                      ``config_context(cache=False)``
``JevAuthError``                                                ``ProviderAuthError`` (any provider)
``JevCostCapError``                                             ``BudgetExceededError`` (any provider)
``.env`` read from the working directory                        environment variables only
``think_reason_learn.core.llms``                                ``think_reason_learn.llm``
``think_reason_learn.core.exceptions``                          ``think_reason_learn.exceptions``
``TRLModel``                                                    ``TRLClassifier``
``random_state``: 42 (RRF, Rule Mining), 0 (Policy Induction)   42 for every model, GPTree and VerifiableRL too
==============================================================  =======================================================================

VerifiableRL has ``fit``, ``predict`` and ``predict_proba`` but no async versions, and no ``verbose``. Reasoned Rule
Mining takes no ``instructions`` and no ``n_examples``. Costs are counted for every model in the price table; set
``LLM(..., price=Price(...))`` for others.

GPTree
~~~~~~

========================================================  ==========================================================
0.2                                                       0.3
========================================================  ==========================================================
``critic_llmc`` and ``use_critic``                        ``critic=`` (the generator by default) and ``use_critic``
``qgen_instr_llmc``                                       removed (never used)
``min_question_candidates``, ``max_question_candidates``  ``max_questions``
``max_node_width``                                        ``max_choices`` (3, as before; the paper uses 2)
``advice()``, ``expert_advice()``                         ``expert_advice=``
``decision_threshold``                                    ``beta``: a number tunes the paper's leaf cut
``fit_iter``, ``resume_fit``, ``stop``                    removed; reruns resume from the answer cache
``view_node``, ``get_node()``, ``get_root_id()``          ``plot()``, ``export_graphviz()``, ``tree_``
``get_questions()``, ``get_training_data()``              ``tree_.to_frame(classes_)``
``get_leaf_proba()``, ``get_leaf_prediction()``           ``apply(X)``, ``predict_proba(X)``
``classes``                                               ``classes_``
``prune_tree(node_id)``                                   ``prune(node_id)``
========================================================  ==========================================================

Random Rule Forest
~~~~~~~~~~~~~~~~~~

===================================================================  =====================================================================
0.2                                                                  0.3
===================================================================  =====================================================================
``max_generated_questions``                                          ``n_prompts`` × ``questions_per_prompt`` (100 by default)
``aggregation_method`` (default ``"elasticnet"``)                    ``combiner`` (default ``"elasticnet"``; ``"vote"`` is the paper's)
``aggregation_max_k``                                                ``max_rules``
``question_scoring_f_beta``, ``aggregation_metric``                  ``beta`` (default 1, so F1 as in 0.2)
``elasticnet_cs``, ``elasticnet_l1_ratios``, ``elasticnet_cv``       0.2's defaults (C of 1, L1 ratio 0.1 or 0.5); ``cv`` (default 3)
``answer_features``                                                  removed; probabilities are used when the answerer gives them
``answer_similarity_func``, ``filter_questions_on_pred_similarity``  ``hamming_threshold``
``add_question``                                                     ``expert_questions=``
``get_questions()``                                                  ``questions_``
``get_answers()``, ``get_answer_probabilities()``                    ``answers(X)``
``predict_founder_level``, ``aggregate_predictions``                 ``predict``, ``predict_proba``
``cost_sensitive`` and its config                                    ``screening_size``, ``screened_questions``
``cross_validate_aggregation``                                       refit with other settings (answers are cached) or ``cross_val_score``
``update_question_exclusion``, ``exclusion_report``                  removed
prompt presets, cumulative memory, semantic filtering                removed
===================================================================  =====================================================================

Policy Induction
~~~~~~~~~~~~~~~~

==================================================  ==========================================================================
0.2                                                 0.3
==================================================  ==========================================================================
``config=WeightTrainerConfig(...)``                 ``beta`` and ``cv``
predicting with per-rule logistic weights (always)  ``predictor="rules"`` (the default); ``"policy"`` is the paper's
``max_policy_length``                               ``max_rules``
``max_gen_batches``                                 ``n_samples``, ``n_rounds``, ``batch_size``; ``memory=True`` for the paper's
``max_samples_as_context``                          ``batch_size`` (10), and ``n_examples`` for the first policy
``get_memory()``                                    ``history_``
``lr``, ``threshold``                               ``combiner_`` (with ``predictor="rules"``)
``validation_result``                               removed
``confirm_requests``                                removed; set ``config_context(max_cost_usd=...)``
==================================================  ==========================================================================

Reasoned Rule Mining
~~~~~~~~~~~~~~~~~~~~

=============================================================  =============================================================================
0.2                                                            0.3
=============================================================  =============================================================================
``config=RRMConfig(...)``                                      ``ensemble_size``, ``use_harsh``, ``harsh_level``, ``beta``, ``random_state``
``ensemble_confidence_threshold`` (0.80)                       ``harsh_gate`` (0.8): samples at or above it get the harsh pass
other ``RRMConfig`` fields (filters, floors, grids)            removed; one logistic regression calibrates both stages
token log probabilities                                        probabilities from decision models or the ensemble's votes
``policy``, ``rules``                                          ``policy_``, ``mined_rules_``
``threshold``, ``weights``                                     ``combiner_``
``validation_result``                                          removed
=============================================================  =============================================================================

VerifiableRL
~~~~~~~~~~~~

===============================================  ============================================================
0.2                                              0.3
===============================================  ============================================================
``config=VerifiableRLConfig(...)``               constructor parameters
``X`` as a dict of arrays                        a 2-D array (``slots`` gives each slot's columns), or a dict
``profiles=``                                    ``slot_texts=`` (per slot, one text per sample)
``LLMActionSupervisor(...)``                     ``supervisor="provider:model"``
``clf_threshold``                                ``reward_threshold``
``predict_threshold``                            ``threshold`` (0.3; ``None`` tunes it for F-beta)
``policy_replay_max``, ``clf_replay_max``        ``policy_replay``, ``clf_replay``
``predict_paths``                                ``decision_paths``
``pretrain_classifier``                          part of ``fit`` (``pretrain_epochs``)
``greedy``                                       removed; prediction is greedy
``pretrained_classifier``, ``from_state_dicts``  removed; ``save`` and ``load``
``action_dim``, ``state_dim``, ``is_fitted``     removed
===============================================  ============================================================

``predict_proba`` has two columns, labels can be any two values, and PyTorch is an extra:
``pip install 'think-reason-learn[rl]'``.

Lambda features
~~~~~~~~~~~~~~~

=====================================  =================================================================
0.2                                    0.3
=====================================  =================================================================
``llm_priority``                       ``generator=``
``temperature``                        ``LLM(model, temperature=...)``
``await generator.generate(...)``      ``generate(...)``, or ``await agenerate(...)``
``GeneratedRule``, ``GeneratedRules``  not exported (the writer's reply format); rules are ``Rule``
=====================================  =================================================================

Saved models
~~~~~~~~~~~~

0.3 cannot load a folder saved by 0.2: ``load`` says so, and the model has to be fitted again. 0.2 stays on PyPI
for its saves. A 0.3 save is one ``model.json`` (VerifiableRL adds its network weights in ``weights.pt``). It records
the save format and the library version, so a later version can read it or say why not, and loading it never runs
code from the file.

Refitting need not pay for the questions again. 0.2's RRF kept its questions in ``questions.parquet``, and Policy
Induction kept its policy in ``policies.parquet``; pass them in, and only the answers are paid for. Reading parquet
needs ``pip install pyarrow``.

.. code-block:: python

    import pandas as pd

    from think_reason_learn.policy_induction import PolicyInduction
    from think_reason_learn.rrf import RRF

    questions = pd.read_parquet("my_0.2_rrf/questions.parquet")["question"]
    rrf = RRF(task="...", n_prompts=0, expert_questions=tuple(questions)).fit(X, y)

    policy = pd.read_parquet("my_0.2_policy_induction/policies.parquet")["policy"]
    policy_induction = PolicyInduction(
        task="...", initial_policy=tuple(policy), n_rounds=0
    ).fit(X, y)

With ``n_prompts=0`` RRF writes no questions, and with ``initial_policy`` and ``n_rounds=0`` Policy Induction uses
the policy as it is; neither needs a generator.
