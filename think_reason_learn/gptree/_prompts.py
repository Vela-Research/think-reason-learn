num_questions_tag = "<number_of_questions>"
INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS = f"""\
You are part of a team of experts that are building a decision tree model (GPTree) for a classification task.
When given a task description, you need to generate clear and detailed instructions template for generating questions for the tree at a Node.
Your instructions template will inform the question generator in your team to generate questions for the tree at a Node.
The number of questions to generate is not known now, so use the tag {num_questions_tag} as a placeholder in your instructions template.
The instructions template should be in a format that is easy for the question generator to understand and generate questions purposely for this task.
The questions must always have definite choices as answers (binary or multi-class).

Generate only the instructions template without prefatory text or labels. Output the raw instructions that will be sent directly to the question generator.
"""
"Instructions for an LLM to generate instructions for generating questions."


QUESTION_ANSWER_INSTRUCTIONS = """\
You are part of a team of experts that are building a decision tree model (GPTree) for a classification task.
When given a question and a sample from the data, your task is to answer the question for the sample per the sample given.
"""
"Instructions for an LLM to answer a question for a sample."


CUMMULATIVE_MEMORY_INSTRUCTIONS = """\
Based on the previous cummulative memory context and what you just learned, \
give a brief advice that can be passed to the next node as cummulative memory context during it's questions generation. \
This should inform the next node about what happened here and the previous nodes and what it should focus on.
"""
"Instructions for an LLM to use cummulative memory context."
