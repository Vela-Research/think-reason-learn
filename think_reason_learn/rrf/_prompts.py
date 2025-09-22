num_questions_tag = "<number_of_questions>"
QUESTION_GEN_INSTRUCTIONS = f"""\
You are part of an intelligent team of LLMs generating YES/NO questions for a 
binary classification task.

Given:
- A task description

Task:
- Generate a prompt/instructions template that instructs a question-generator LLM
  to generate {num_questions_tag} YES/NO distinctive questions that are discriminative 
  for the task.
- Let the question-generator LLM understand the task and generate questions that are
  discriminative for the task.
- Do not restrict the question-generator LLM's creativity. Let it decide what to ask
  based on the clear detailed task description you give in this prompt template.
- Note: Questions are generated through multiple LLM calls. The template should 
  instruct generators to produce cumulative memory - a high-level summary of 
  insights from questions generated so far. This guides future generators on 
  promising directions without constraining creativity. Include instructions for 
  building upon previous cumulative memory rather than starting fresh each time.

Return:
- Only the prompt/instructions template text that will be sent to the question
  generator, and it must contain the placeholder {num_questions_tag} where the
  number of questions will be substituted later.
"""
"Instructions for generating YES/NO questions for a binary classification task."

CUMULATIVE_MEMORY_INSTRUCTIONS = """\
Using the prior cumulative memory (if any) and what was just learned at this generation,
write a brief note for the next generation. Do not discard the prior cumulative memory
but build on it.

Guidelines:
- Summarize key observations and outcomes so far at a high level, and provide gentle
  hints or priorities for the next generation.
- Treat this as guidance, not a constraint; avoid prescriptive language.
- Do not repeat raw samples or sensitive details; avoid long quotes.
- Keep it concise (1-5 sentences).
"""
"Instructions for producing concise, non-restrictive cumulative memory."

QUESTION_ANSWER_INSTRUCTIONS = """\
You are an answering agent in a RRF classification pipeline.

Given:
- A question,
- One sample (text),

Task:
- Answer, YES/NO question based on the sample.
- Be deterministic.

Output:
- Return only the chosen answer string—no explanations, punctuation, or extra text.
"""
"Instructions for answering a YES/NO question based on a sample."