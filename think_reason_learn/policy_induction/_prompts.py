max_policy_num_tag = "<max_policy_length>"

POLICY_GEN_INSTRUCTIONS = f"""\
You are part of an intelligent team of LLMs collaborating on a binary
classification task.

Given:
- A task description.

Task:
- Generate a prompt/instructions template that will be sent to another LLM
  (the policy generator).
- This prompt template should instruct the policy-generator LLM to
  **enrich existing policies** for the binary classification task.
- The policy-generator LLM will receive:
  1. The task description.
  2. A set of existing policies from previous rounds.
  3. Newly collected data samples with their binary labels.
- The goal of the policy-generator LLM is not to replace or discard existing
  policies, but to expand and enrich them — by incorporating insights from new
  data, clarifying ambiguous logic, generalizing existing decision rules, and
  introducing new policies when novel patterns or principles emerge.

- The policies generated should be:
  1. Generalizable: extract broader patterns, not case-specific details.
  2. Transferable: focus on underlying signals that apply beyond single examples.
  3. Concise: keep each policy clear and actionable in one sentence.
  4. Non-overfitting: avoid policies too specific to individual samples.
  5. Bounded: do not exceed {max_policy_num_tag} total policies.

- The resulting prompt should provide high-level guidance on what the policy
  generator should consider and how to approach refinement, without prescribing
  exact rules or features.
- Ensure the template includes the placeholder {max_policy_num_tag} (not any
  exact number), which will be dynamically substituted at runtime.

Return:
- Only the text of the prompt/instructions template for the policy-generator LLM.
"""

POLICY_PREDICT_INSTRUCTIONS = """\
You are a deterministic classification agent.

Given:
- A task description
- A policy
- A single sample (text)

Objective:
Classify the sample as either "YES" or "NO" according to the policy
and the task description.

Requirements:
- Base your decision strictly on the given policy and sample content.
- Be deterministic: the same input must always yield the same output.
- Do not explain your reasoning or include any punctuation.

Output format:
Return ONLY ONE WORD: YES / NO.
"""

POLICY_PREDICT_BATCH_INSTRUCTIONS = """\
You are a deterministic classification agent.

Given:
- A task description
- A numbered list of policies, each labelled with a unique integer id
- A single sample (text)

Objective:
For EVERY policy in the list, classify the sample as either "YES" or "NO"
according to that policy and the task description.

Requirements:
- Judge each policy INDEPENDENTLY. A policy's answer must not be influenced
  by any other policy in the list, by their order, or by how many YES or NO
  answers you have already produced. Treat each one as if it were the only
  policy you had been given.
- Base each decision strictly on that policy, the sample content, and the
  task description.
- Be deterministic: the same input must always yield the same output.
- Return exactly one answer for every policy id — no more, no fewer.
- Echo each policy's id exactly as it appears in the list. Do not renumber
  or invent ids.
- Do not explain your reasoning.

Output format:
One entry per policy, each carrying that policy's id and its answer (YES / NO).
"""
