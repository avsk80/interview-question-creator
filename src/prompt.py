prompt_template = """
You are an expert at creating questions related to Big Data Analytics.

Create a graduate level multiple question on the Big Data Analytics from the context provided, with at least 4 or more answer options. 
More than 1 correct answers are not possible. Do not give the correct answer and any explanation on why the correct answer(s) is/are true.

Below is the context for asking questions about the topic:

------------
{text}
------------

Strictly follow the below output format enclosed in <>:
Example output:
<
Question 1: What is the mathematical definition of a Dynamic Bayesian Network (DBN)?
a) A DBN is a probabilistic graphical model that represents a set of random variables and their conditional dependencies over time.
b) A DBN is a type of recurrent neural network that uses Bayesian inference to model temporal dependencies.
c) A DBN is a type of decision tree that uses Bayesian statistics to make predictions.
d) A DBN is a type of clustering algorithm that uses Bayesian methods to group similar data points.

Question 2: ...?
a) ...
b) ...
c) ...
d) ...
>

Output the above format as a python list of JSON objects, where each element in the JSON contains 2 key-value pairs - question and answer_choices of the form:
```
1) key 1 = question , value 1 = question as a string
2) key 2 = answer_choices, value 2 = array of answer_choices (value is a list of answer choices) 
```

Your response should STRICTLY follow the above structure within enclosed quotes ```.

Note:
1) Create exactly 2 questions that will prepare the student for their tests.
2) Make sure not to lose any important information.
3) Ignore referring to the name of professor or the lecture slide lesson names or the course names in the questions.
4) The questions should solely focus on testing the technical concepts covered in the course material.
5) The questions should be framed in such a way that the students can gain knowledge from these questions and apply them as a future data scientist or a data engineer.

"""

refine_template = ("""
You are an expert at creating multiple answer choices for practice questions based on Big Data Analytics.
Your goal is to help a student prepare for a test.
We have received some practice questions to a certain extent: {existing_answer}.
The questions and the respective answer choices are python list of JSON objects, in the following format enclosed in ```. 
Parse it accordingly:

```
1) key 1 = question , value 1 = question as a string;
2) key 2 = answer_choices, value 2 = array of answer_choices (value is a list of answer choices) 
```
                   
Note: Do not alter the above input format. Your response should STRICTLY follow the above structure in enclosed quotes ```.                  

We have the option to refine the existing questions. Always create 4 answer choices (A, B, C, D) as the multiple choices for each of the questions.
Those choices should be logical, intuitive, and directly related to the question.
Refine the questions and or the options as and when needed to make them very lucid.                      
Modify the questions (only if necessary) with some more context below.
------------
{text}
------------


Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)