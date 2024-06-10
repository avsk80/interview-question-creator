import os
from dotenv import load_dotenv

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


llm_pipeline(file_path="/home/learning/gen_ai/one_neuron/interview-question-creator/data/SDG.pdf")