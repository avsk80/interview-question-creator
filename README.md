# interview-question-creator

## Commands for setup

#### Git commands used:

```bash
1. git clone https://github.com/avsk80/interview-question-creator.git

2. git add .

3. git commit -m "updated README"

4. git push origin main
```

#### Python Environment Setup:

```bash
1. conda create -n interview python=3.10 -y

2. conda activate interview

3. pip install -r requirements.txt

```

#### Technologies used:
**Frontend**: HTML, CSS, JS <br>
**Backend**: FastAPI <br>
**ML stack**: Langchain, Python, Jupyter Notebook <br>

#### Procedure to run the code base:

```bash
Python app.py
```

#### To test functions locally use the script: 
``` bash
test/test-unit.py
```

#### UI:
1. File upload functionality - PDF and text. <br>
2. Upload PDF for generating Questions and Answers. <br>
3. Optionally upload a text file if you already have predetermined questions to get LLM-generated answers. <br>

![UI](https://github.com/avsk80/interview-question-creator/blob/main/images/quiz-4.png)

#### View the uploaded PDF:
![view-pdf](https://github.com/avsk80/interview-question-creator/blob/main/images/quiz-2.png)

#### LLM-generated Q & A:
![QA](https://github.com/avsk80/interview-question-creator/blob/main/images/quiz-3.png)

#### FAQs:
1. How good are the generated questions? Is this App reliable? <br>
Ans: I have used a refine chain strategy to get LLM-generated questions. I use 2 prompts - a base prompt that generates a list of questions as a response from LLM, and then the response is fed as context to a refine prompt, that enhances the questions and multiple-choice answers.

2. What is the essence of this whole project from technical and business standpoints? <br>
Ans: <br>
Business view: <br>
This App can help professors, and TAs generate quiz questions and answers to test students. This can significantly reduce their workload and also help them set more creative and trick questions.


Tech view: <br>
I followed a typical RAG architecture in this project.<br>

Case A: If a list of questions is uploaded from the UI in a text file. Then, use the pdf as context and generate answers from the LLM. <br>

Case B: If no text file is uploaded, then generate both questions and answers from the pdf. <br>

For questions:

Storage:<br>
1) First, I get the pdf as context and create document chunks. <br>
2) Next, I numericalize them using embeddings (OpenAI in my case). <br>
3) Finally, I store the embeddings in a vector DB (Chroma in my case). <br>

Retrival and Generation:<br>
1) Given a prompt and context, we extract relevant documents from the vector DB and feed it to the LLM in refine fashion to generate a list of questions.
 query (question)

For answers: <br>
1) Pass the context to the LLM in the same fashion as above and generate LLM-driven answer explanations. <br>

3. What parameters can be changed in this App to suit specific user needs? <br>
Ans: <br>
1. To be more creative one can always play with the temperature parameter. <br>
2. Alter the number of questions to be generated by the LLM. <br>

4. Further enhancements? <br>
Ans: <br>
1. Agents can be used rather than RAG so that LLM can refer internet to give more creative explanations. <br>
2. To tackle more diverse question types like multi-correct answers, coding questions, short answer texts etc.
