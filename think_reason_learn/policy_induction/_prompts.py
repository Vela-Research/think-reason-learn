max_policy_num_tag = "<max_policy_length>"

POLICY_GEN_INSTRUCTIONS = f"""\
You are part of an intelligent team of LLMs collaborating on a binary 
classification task.

Given:
- A task description.

Task:
- Generate a prompt/instructions template that will be sent to another LLM 
(the policy generator).
- This prompt template should instruct the policy-generator LLM to **refine 
and extend existing policies** for the binary classification task.
- The policy-generator LLM will receive:
  1. The task description.
  2. A set of existing policies from previous rounds.
  3. Newly collected data samples with their binary labels.
- The goal of the policy-generator LLM is **not to replace or discard** existing 
policies, but to **improve them** by incorporating insights from the new data,
clarifying ambiguous logic, and generalizing existing decision rules.
  
- The policies generated should be:
1. Generalization: Extract broader patterns rather than specific 
details from this example
2. Refinement: Improve existing policies to be more universally applicable 
across different startups
3. Focus on transferable insights: Identify underlying success 
factors that apply beyond this specific case
4. Maintain conciseness: Keep policies clear and actionable (max 20 items)
5. Avoid overfitting: Don't create policies that are too specific 
to this single example
6. Do not exceeding {max_policy_num_tag} rows.

- The resulting prompt should provide high-level guidance on 
what the policy generator should consider and how it should approach 
refinement, without prescribing exact rules or data features to focus on.
- Your task is to produce only the **prompt template** that guides the 
policy-generator LLM in this refinement process.
- Ensure the template you produce includes the placeholder {max_policy_num_tag},
  which represents the maximum allowed policy length and will be 
  dynamically substituted later.

Return:
- Only the text of the prompt/instructions template that will be sent to the
  policy-generator LLM.
"""
"Instructions for generating policies for a binary classification task."


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
"Instructions for prediction based on a sample and a policy."
