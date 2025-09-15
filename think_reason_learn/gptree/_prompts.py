num_questions_tag = "<number_of_questions>"
INSTRUCTIONS_FOR_GENERATING_QUESTION_GEN_INSTRUCTIONS = f"""\
You are part of a team of experts that are building a decision tree model (GPTree) for a classification task.
When given a task description, you need to generate clear and detailed instructions template for generating questions for the tree at a Node.
Your instructions template will inform the question generator in your team to generate questions for the tree at a Node.
The number of questions to generate is not known now, so use the tag {num_questions_tag} as a placeholder in your instructions template.
The instructions template should be in a format that is easy for the question generator to understand and generate questions purposely for this task.
"""
"Instructions for an LLM to generate instructions for generating questions."


QUESTION_ANSWER_INSTRUCTIONS = """\
You are part of a team of experts that are building a decision tree model (GPTree) for a classification task.
When given a question and a sample from the data, your task is to answer the question for the sample per the sample given.
"""
"Instructions for an LLM to answer a question for a sample."
