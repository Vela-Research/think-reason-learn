"""LLM prompts for Reasoned Rule Mining, as module-level constants.

All instructions are task-agnostic: the per-task description and the data are
passed in the ``query`` (mirroring ``policy_induction``), so these stay static.
The original VCBench script hard-coded founder-specific guidance; that domain
content belongs in the caller's ``task_description``, not in the library.
"""

REASONING_INSTRUCTIONS = """\
You are a careful analyst. You are given a binary classification task, a single
sample, and that sample's KNOWN true label (YES or NO). Explain, step by step
(chain-of-thought), the key reasons -- grounded in the concrete attributes of
the sample -- why this sample has that label. Focus on transferable,
generalizable signals rather than incidental details. Be concise.
"""
"Instruction for the per-sample reasoning-log stage."

RULE_EXTRACTION_INSTRUCTIONS = """\
Convert the following reasoning log into a SINGLE structured logical rule in the
format:
IF <conditions> THEN label = <YES or NO>.

Requirements:
1. Generate ONLY ONE rule.
2. Use ONLY 'YES' or 'NO' as the outcome.
3. Base the conditions only on attributes of the sample in the reasoning log.
4. Keep it concise and generalizable.

Return only the rule text.
"""
"Instruction for converting a reasoning log into one IF-THEN rule."

POLICY_COMPILATION_INSTRUCTIONS = """\
You are compiling a decision policy for a binary classification task from a set
of extracted IF-THEN rules. Synthesize the YES-rules and the NO-rules into a
single, coherent decision policy. Output exactly two lines:
For YES: IF <conditions> THEN label = YES
For NO: IF <conditions> THEN label = NO
"""
"Instruction for synthesizing extracted rules into one decision policy."

VOTE_INSTRUCTIONS = """\
You are a deterministic classification agent. Given a task description, a
decision policy, and a single sample, decide the label using the policy as the
rubric. Base your decision strictly on the sample content and the policy.
Respond with only YES or NO.
"""
"Instruction for a single ensemble vote."

HARSH_INSTRUCTIONS = {
    "light": """\
You are a critical but fair analyst performing a re-evaluation of a sample that
was initially predicted YES. Re-examine the prediction using the same decision
policy with a balanced perspective that recognizes both strengths and
weaknesses. Only predict YES if the sample clearly meets the policy criteria.
Respond with only YES or NO.
""",
    "moderate": """\
You are an analyst performing a careful re-evaluation of a sample initially
predicted YES. Identify samples with genuine support for YES while filtering out
those that do not meet the policy's standards. When in doubt, lean towards NO,
but weigh the overall evidence. Respond with only YES or NO.
""",
    "strict": """\
You are a highly critical and demanding analyst performing a harsh
re-evaluation of a sample initially predicted YES. Apply the decision policy
with maximum scrutiny. Only predict YES if the sample is exceptional with no
significant weaknesses. Respond with only YES or NO.
""",
}
"Instructions for harsh re-evaluation, keyed by severity level."

MEMORY_SUMMARY_INSTRUCTIONS = """\
Summarise the key recurring patterns and signals from the reasoning logs so far
into a short, running set of notes that will help analyse further samples. Merge
with the previous summary if one is given; keep it concise (a few bullet
points). Return only the updated summary.
"""
"Instruction for the rolling reasoning-memory summary."
