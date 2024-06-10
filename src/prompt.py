prompt_template = """
You are an expert at creating questions based on sustainable development goals.
Your goal is to prepare a student for their exam.
You do this by asking questions about the text below:

------------
{text}
------------

Create 10 questions that will prepare the student for their tests.
Make sure not to lose any important information.

QUESTIONS:
"""

refine_template = ("""
You are an expert at creating practice questions based on sustainable development goals.
Your goal is to help a student prepare for a test.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)
