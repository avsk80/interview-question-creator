import os
from dotenv import load_dotenv

"""
from langchain.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings

from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from src.prompt import prompt_template, refine_template




# OPENAI Authentication
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


# file preprocessing --> ertract content from pdf as docs, generate question chunks and answer chunks and return the split ques, ans chunks
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()

    ques_gen = ""

    for page in data:
        ques_gen += page.page_content

    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size = 10000,
        chunk_overlap = 200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(ques_gen)

    document_ques_gen = [ Document(t) for t in chunks_ques_gen ]

    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size = 1000,
        chunk_overlap = 100
    )

    document_ans_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_ans_gen

def llm_pipeline(file_path):

    document_ques_gen, document_ans_gen = file_preprocessing(file_path)

    llm_ques_gen_pipeline = ChatOpenAI(temperature=0.3,
                                        model='gpt-3.5-turbo')
    
    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=['text'])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(template=refine_template, input_variables=['existing_answer', 'text'])

    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline,
                                          chain_type="refine",
                                          verbose=True,
                                          question_prompt = PROMPT_QUESTIONS,
                                          refine_prompt = REFINE_PROMPT_QUESTIONS
                                          )
    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(document_ans_gen, embeddings)

    llm_answer_gen = ChatOpenAI(temperature = 0.1, 
                                model = "gpt-3.5-turbo")

    ques_list = ques.split("\n")

    filtered_ques_list = [ element for element in ques_list if element.endswith('?') or element.endswith('.') ]

    answer_gen_chain = RetrievalQA.from_chain_type(
        llm = llm_answer_gen,
        verbose = True,
        chain_type = "stuff",
        retriever = vector_store.as_retriever()
    )

    return answer_gen_chain, filtered_ques_list


# llm_pipeline(file_path="/home/learning/gen_ai/one_neuron/interview-question-creator/data/SDG.pdf")
"""


from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.document_loaders import TextLoader

from src.prompt import prompt_template, refine_template

# OPENAI Authentication
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def generate_ques_list(context):

    print("Generating list of questions")

    ques_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    ques_splits = ques_text_splitter.split_documents(context)

    PROMPT_QUESTIONS = PromptTemplate.from_template(prompt_template)
    REFINE_PROMPT_QUESTIONS = PromptTemplate.from_template(refine_template)

    ques_llm = ChatOpenAI(temperature=0.3, model='gpt-3.5-turbo')

    ques_gen_chain = load_summarize_chain(llm=ques_llm,
                                          chain_type="refine",
                                          verbose=True,
                                          question_prompt = PROMPT_QUESTIONS,
                                          refine_prompt = REFINE_PROMPT_QUESTIONS,
                                          input_key="input_documents",
                                          output_key="output_text"
                                          )
    
    result = ques_gen_chain.invoke({"input_documents": ques_splits}, return_only_outputs=True)

    result_list = result['output_text'].split('\n\n')
    question_list = [ question for question in result_list ]

    print("Here is the question list!!!!!!!!!!!!")

    print(question_list)

    return question_list


def llm_pipeline(file_path, txt=None):
    """
    Returns - Answer generation chain and question list
    """

    pdf_loader = PyPDFLoader(file_path=file_path)
    # Context for generating and answering questions
    context = pdf_loader.load()

    # Get question list

    if txt:
        """
        If txt file is passed just parse it to get the list of questions.

        Logic:
        1. Use Text based loader
        2. Parse the documents and structure it to feed the questions to an LLM in the format [q1 <Options-A,B,C,D>, q2 <Options-A,B,C,D>, q3 <Options-A,B,C,D>, ....]
        """
        loader = TextLoader(file_path=txt)
        txt_data = loader.load()

        question_list = txt_data[0].page_content.split("\n\n")

    else:
        """
        If no txt file is passed, use the given context to create question list.

        Logic:
        1. Use PDF loader
        2. Chunk the text and create embeddings
        3. Use refine Strategy to pass the info to an LLM and get documents of questions
        4. Parse the output to create list of questions. 
        5. Structure the questions to feed them to an LLM in the format [q1 <Options-A,B,C,D>, q2 <Options-A,B,C,D>, q3 <Options-A,B,C,D>, ....]
        """
        question_list = generate_ques_list(context)

    # Generate Answers from the LLM by passing questions one by one and eventually 

    """
    1. Create an LLM for generating Answers
    2. Split the context and embed the chunks to a vectordb
    3. create a chain, and invoke it by passing the questions one by one 
    """

    ans_llm = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo')

    ans_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=400, add_start_index=True
    )

    ans_splits = ans_text_splitter.split_documents(context)

    vectorstore = Chroma.from_documents(ans_splits, OpenAIEmbeddings())

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Keep the answer as concise as possible, provide a simple example so that students can understand the concepts intuitively. 
    {context}
    Question: {question}

    Output format should be of the form:
    ```
    Answer: Option C
    Explanation: ..... (including and example whenever applicable).
    ```
    Follow the above structure of output STRICTLY.
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    answer_gen_chain = RetrievalQA.from_chain_type(
        llm = ans_llm,
        verbose = True,
        chain_type = "stuff",
        retriever = vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return answer_gen_chain, question_list

