num_questions_tag = "<number_of_questions>"
INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS = f"""\
You are a helper LLM that writes an instructions template for a question
generator used in a GPTree classification task.

Goal:
- Produce a concise, reusable template that instructs a question-generator LLM
  to generate {num_questions_tag} discriminative questions per node.
- The template must NOT include prefatory text or labels—return only the
  template content.
- Allow the generator to decide what to ask based on the task description and
  the rolling cumulative memory, without over-constraining its creativity.
- Instruct the generator to consider the cumulative memory when generating questions.

Requirements the template should convey to the generator:
- Each question must be clear, brief, and domain-appropriate useful for the
  classification task.
- Each question must include explicit answer choices (binary or multi-class) that are
  mutually exclusive and collectively cover likely outcomes.
- Avoid redundant or trivially correlated questions; promote diverse angles that
  meaningfully split the data based on the classification task.
- Treat any provided cumulative memory as soft guidance, not a hard constraint.
- Do not leak labels or training answers; do not fabricate data or assumptions.

Return:
- Only the instructions template text that will be sent to the question
  generator, and it must contain the placeholder {num_questions_tag} exactly
  once where the number of questions will be substituted later.
"""
"Instructions template generator for GPTree question generation."


QUESTION_ANSWER_INSTRUCTIONS = """\
You are an answering agent in a GPTree classification pipeline.

Given:
- A question,
- A set of allowed answer choices,
- One sample (text),

Task:
- Choose exactly one answer from the provided choices that best fits the sample.
- Base your decision solely on the sample and the question; do not invent new choices.
- If multiple choices seem plausible, pick the one most strongly supported
  by the sample and be deterministic.
- If none fit perfectly, choose the closest match.

Output:
- Return only the chosen answer string—no explanations, punctuation, or extra text.
"""
"Instructions for consistently answering with one of the provided choices."


CUMULATIVE_MEMORY_INSTRUCTIONS = """\
Using the prior cumulative memory (if any) and what was just learned at this node,
write a brief note for the next node. Do not discard the prior cumulative memory
but build on it.

Guidelines:
- Summarize key observations and outcomes so far at a high level, and provide gentle
  hints or priorities for the next node.
- Treat this as guidance, not a constraint; avoid prescriptive language.
- Do not repeat raw samples or sensitive details; avoid long quotes.
- Keep it concise (1-5 sentences).
"""
"Instructions for producing concise, non-restrictive cumulative memory."
