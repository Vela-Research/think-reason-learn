Policy decisions
----------------

Some decisions have more than two outcomes, and the outcomes come in order: a gift the recipient will not like, may
like, or will love. ``PolicyClassifier`` makes decisions like these, with 2 to 4 ordered outcomes, from a written
policy, from labelled cases, or from both. Every prediction comes with points by criterion that add up to its score.

Write the policy
~~~~~~~~~~~~~~~~

A policy has its outcomes, lowest first, and the criteria it weighs. Give each criterion an id of your own and a kind:
``"positive"`` if meeting it points to a higher outcome, ``"negative"`` if to a lower one, ``"neutral"`` if either.
Write each outcome's meaning as a statement about a case, since the checks quote it.

.. code-block:: python

   from think_reason_learn.policy import Criterion, Outcome, Policy, PolicyClassifier

   policy = Policy(
       text="Will the recipient love this gift?",
       outcomes=(
           Outcome("NO", "the recipient will not like the gift"),
           Outcome("MAYBE", "the recipient may like the gift"),
           Outcome("YES", "the recipient will love the gift"),
       ),
       criteria=(
           Criterion("C1", "The gift matches a hobby the recipient has.", "positive"),
           Criterion("C2", "The recipient already owns something like the gift.", "negative"),
           Criterion("C3", "The gift is handmade.", "positive"),
       ),
   )

The text may be empty when criteria are given. A policy with text and no criteria is judged as a whole.

Fit
~~~

Give labels where you have them, ``None`` where you do not, and a teacher to label the rest. The teacher is a strong
reasoning model that applies the policy to one case at a time. Your labels always win: the teacher never labels a
case you labelled. It does judge a sample of them, to measure how often it agrees with you (see the report below).

.. code-block:: python

   X = [
       "A knitting kit for a friend who knits every weekend. Handmade by a local artist.",
       "A second coffee grinder for someone who already has one.",
       "A mass-produced mug with a joke on it.",
       # ...
   ]
   y = ["YES", "NO", None]  # None: the teacher labels it

   clf = PolicyClassifier(
       policy,
       answerer="jev",                           # answers the yes/no checks
       teacher="anthropic:claude-sonnet-5-5",    # labels the cases with no label
       writer="openai:gpt-5",                    # optional: writes new checks
   )
   clf.estimate_cost(X, y)    # calls, tokens and dollars; no model is called
   clf.fit(X, y)

   clf.predict(["A climbing chalk bag for a climber."])
   clf.predict_proba(X)       # one column per outcome, in classes_ order
   clf.outcomes_              # the outcomes, lowest first

If one outcome has more than 90% of the training labels, ``fit`` warns: there is little to learn the others from,
and the policy may be too strict or too loose for the cases. As in scikit-learn, ``classes_`` holds the outcomes sorted, and ``predict_proba``'s columns follow it, so
scikit-learn's scorers and ``cross_val_predict`` read them right; ``outcomes_`` holds them in the policy's order. With
labels for every case no teacher is needed; with no labels at all, call ``fit(X)``. To estimate a fit with no
labels, pass ``estimate_cost(X, [None] * len(X))``. ``max_cost_usd=`` caps each fit and predict, like
``config_context(max_cost_usd=...)``.

What fit does
~~~~~~~~~~~~~

1. **Labels.** Your labels are kept; the teacher labels the rest and says which criteria it could not judge from a
   case's text. A criterion it could not judge on more than 80% of its cases is reported and left out. With
   ``teacher_draws=2`` it judges each case twice: a case judged two ways weighs half, and the report gives how often
   the two draws agreed.
2. **Checks.** Each criterion becomes yes/no checks by the ``templates``: ``"reach"`` asks, once per boundary,
   whether judging by that criterion alone the case reaches the outcome; ``"applies"`` asks whether the criterion bears
   on the case; ``"met"`` asks whether the case meets it; and ``"policy"`` asks the first of these for the whole policy,
   once per boundary. All four are used by default. Checks of your own go in ``checks=``, asked as written.
3. **Written checks.** With a writer, a model reads the cases the current checks get wrong, out of fold, with the
   teacher's reasons, and writes new concrete checks. ``writer_rounds`` sets how many rounds; one by default.
4. **Weights.** The answerer answers every check for every labelled case, and an ordinal model weighs the answers, one
   weight per check. With ``signs="criteria"`` a positive check may only raise the outcome and a negative one only
   lower it, so the reasons stay readable. The model predicts the most probable outcome; ``decision="balanced"``
   predicts rare outcomes more often instead.

Explain a prediction
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   [why] = clf.explain(["A climbing chalk bag for a climber."])
   why.outcome        # the predicted outcome
   why.points         # the case's score
   why.reasons        # points by criterion id, largest effect first
   why.cutoffs        # the score from which the model predicts each outcome or a higher one
   why.unassessed     # criteria that could not be judged for this case

``why.baseline + sum(why.reasons.values()) == why.points``, exactly. The baseline is the score of a case whose every
answer is at its training average.

Read the report
~~~~~~~~~~~~~~~

``clf.report_`` says how well the model agrees with its labels on cases it did not learn from (out of fold): the
share exactly right, balanced accuracy, quadratic kappa and the AUC of each boundary, for your labels and the
teacher's apart. It also gives each source's outcome shares, the teacher's consistency, and the criteria left out. When there are
both your labels and a teacher, the teacher also judges up to ``teacher_check`` (50) of your cases, only to measure how
often it gives your label: ``report_["policy_vs_labels"]`` has the share it matches, the quadratic kappa and how many
it judged. Your labels stay as they are, and these verdicts are not learned from.
Written checks were written from these same cases, so with a writer the report is optimistic: keep some labelled
cases apart to measure it. ``clf.labels_`` has each training case's outcome, source, confidence and reason, and
``clf.checks_`` each check with its points. A check that answered the same for nearly every training case
(``flat``) gets no weight: it tells the cases apart too little to trust.

Change the weights without calling a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The answers from the fit are kept, so ``refit`` weighs them again at no cost: with corrected labels, with some
criteria or checks muted (no weight), or with some locked (their weight kept as fitted).

.. code-block:: python

   clf.refit(mute=["C3"])
   clf.refit(labels=corrected)  # one outcome per training case, in labels_ order

Saving works like the other models. A teacher that is a function of your own cannot be saved, so the saved model has
none; it still predicts and refits.

The steps on their own
~~~~~~~~~~~~~~~~~~~~~~

Each step can be used alone, through the same cache, retries and cost cap.

.. code-block:: python

   from think_reason_learn.core import fit_ordinal
   from think_reason_learn.policy import answer_checks, label_cases, write_checks

   labels = label_cases(policy, X, "anthropic:claude-sonnet-5-5", y=y)
   answers = answer_checks(X, ["The gift is handmade."], "jev")  # NaN where unanswered
   codes = labels["outcome"].map({"NO": 0, "MAYBE": 1, "YES": 2}).to_numpy()
   scorer, out_of_fold = fit_ordinal(answers, codes, n_outcomes=3)
   scorer.proba(answers)  # one column per outcome, lowest first

``label_cases`` can also top up an outcome with few labels from a pool of more cases (``pool=``,
``min_per_outcome=``), and records the sampling weights that undo the extra share it gives that outcome. Pass them
on to the fit:

.. code-block:: python

   labels = label_cases(policy, X, "anthropic:claude-sonnet-5-5", pool=more, min_per_outcome=30)
   labels = labels[labels["outcome"].notna()]  # cases the teacher could not judge
   clf = PolicyClassifier(policy, answerer="jev")  # no teacher: the labels are made
   clf.fit(labels["case"], labels["outcome"], sample_weight=labels["weight"])

The fit takes these labels as given (``source`` ``"gold"`` in ``labels_``), so this classifier has no teacher. With
one, ``teacher_check`` would ask the same teacher about its own labels and report it agreeing with itself.
