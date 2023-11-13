import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

# Load configuration file
with open("../conf/config.yaml", "r") as file:
    config = yaml.safe_load(file)
print(config)


# Load environment variables (including OpenAI API Key)
dotenv_path = Path("../.env")
if not load_dotenv(dotenv_path):
    print(
        "Could not load .env file or it is empty. Please check if it exists"
        " and is readable."
    )
    exit(1)


def create_embeddings_and_vectorstore(file_path):
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    # Chunk and Embeddings
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=600,
    #     chunk_overlap=300,
    #     separators=["\n\n", "\n", " ", ""],  # adjust these as necessary
    # )

    texts = text_splitter.split_documents(pages)

    if config["embedding_llm"] == "OpenAI":
        embeddings = OpenAIEmbeddings()
    elif config["embedding_llm"] == "Llama2-7B (4bit)":
        embeddings = LlamaCppEmbeddings(
            model_path="../models/llama-2-7b-chat.Q4_K_M.gguf",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=2000,
        )
    elif config["embedding_llm"] == "Llama2-7B (8bit)":
        embeddings = LlamaCppEmbeddings(
            model_path="../models/llama-2-13b-chat.Q5_K_M.gguf",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=2000,
        )
    elif config["embedding_llm"] == "all-mpnet-base-v2":
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )  # , model_kwargs={"device": "cuda"})
    elif config["embedding_llm"] == "all-MiniLM-L6-v2":
        emb_model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=emb_model,
            # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME') # TODO Look into cache_folder param
        )
    elif config["embedding_llm"] == "instructor-large":
        emb_model = "hkunlp/instructor-large"
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=emb_model,
            # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME') # TODO Look into cache_folder param
        )
    else:
        print("Unavailable embedding_llm. Please check config yaml")

    # Vector Store
    db = Chroma.from_documents(documents=texts, embedding=embeddings)
    return db


selected_model = config["llm"]
if selected_model == "GPT-3.5-turbo":
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
elif selected_model == "GPT-4":
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
elif selected_model == "Llama2-7B (4bit)":
    llm = LlamaCpp(
        model_path="../models/llama-2-7b-chat.Q4_K_M.gguf",
        verbose=False,
        n_ctx=4048,
        temperature=0,
        max_tokens=0,
    )  # n_ctx is number of tokens used for context, and max_tokens is length of response #Source: https://swharden.com/blog/2023-07-29-ai-chat-locally-with-python/
elif selected_model == "Llama2-7B (8bit)":
    llm = CTransformers(
        model="../models/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={"max_new_tokens": 512, "temperature": 0.01},
    )
elif selected_model == "Llama2-13B (5bit)":
    llm = LlamaCpp(
        model_path="../models/llama-2-13b-chat.Q5_K_M.gguf",
        verbose=False,
        n_ctx=4048,
        streaming=False,
        temperature=0.1,
        n_gpu_layers=1,
        n_batch=512,
    )
else:
    print("Unavailable summarization llm. Please check config yaml")

time_start = time.time()
db = create_embeddings_and_vectorstore(config["Input document"])
time_elapsed = time.time() - time_start

print(f"Embedding time: {time_elapsed:.2f} sec")


default_rag_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:
"""

llama_rag_template = """[INST]<<SYS>> Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.<</SYS>>
    Question: {question}
    Context: {context}
    Answer: [/INST]
"""
# NOTE: Llama2 requires a different prompt template with [INST] and <<SYS>> tags
# Interestingly, I don't think RetrievalQA has a ConditionalPromptSelector to auto switch to llama prompt
# Code currently uses default question_answering prompt (https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/retrieval_qa/prompt.py)
default_rag_prompt = PromptTemplate.from_template(default_rag_template)
llama_rag_prompt = PromptTemplate.from_template(llama_rag_template)

rag_prompt_selector = ConditionalPromptSelector(
    default_prompt=default_rag_prompt,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), llama_rag_prompt)],
)

system_prompt = rag_prompt_selector.get_prompt(llm)

# Initialise RetrievalQA Chain
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(
        search_kwargs={"k": 2}
    ),  # search_type="mmr"),#search_kwargs={"k":3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": rag_prompt_selector.get_prompt(llm)},
)

prompt = "tell me a joke"
time_start = time.time()
response = chain({"query": prompt})
# response = chain(prompt)
time_elapsed = time.time() - time_start
print(response["result"])
print(f"Response time: {time_elapsed:.2f} sec")

for i, source_doc in enumerate(response["source_documents"]):
    print(f"### Source Document {i+1}")
    print(source_doc.page_content)
    print(f'Page {source_doc.metadata["page"]}')
    print(f'Source file: {source_doc.metadata["source"]}')
