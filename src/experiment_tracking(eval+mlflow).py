import os
import time
from pathlib import Path

import pandas as pd
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
from ragas import evaluate
from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import answer_relevancy
from ragas.metrics import context_precision
from ragas.metrics import context_recall
from ragas.metrics import faithfulness


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

# prompt = "tell me a joke"
# time_start = time.time()
# response = chain({"query": prompt})
# time_elapsed = time.time() - time_start
# print(response["result"])
# print(f"Response time: {time_elapsed:.2f} sec")

# for i, source_doc in enumerate(response["source_documents"]):
#     print(f"### Source Document {i+1}")
#     print(source_doc.page_content)
#     print(f'Page {source_doc.metadata["page"]}')
#     print(f'Source file: {source_doc.metadata["source"]}')

##############################################
# Evaluation

# Configure evaluation LLMs
# Ragas uses gpt3.5 by default - it's possible to change LLM for metrics
faithfulness.llm.langchain_llm = ChatOpenAI(
    model="gpt-3.5-turbo", request_timeout=120
)
context_precision.llm.langchain_llm = ChatOpenAI(
    model="gpt-3.5-turbo", request_timeout=120
)
# answer_relevancy.llm.langchain_llm = ChatOpenAI(model="gpt-3.5-turbo", request_timeout=120)
context_recall.llm.langchain_llm = ChatOpenAI(
    model="gpt-3.5-turbo", request_timeout=120
)

# Import Evaluation dataset
df = pd.read_csv("../data/batman_eval_simple.csv")
df = df.head(2)
eval_questions = df["question"].values.tolist()
eval_answers = df["answer"].values.tolist()
# Create examples using question-answer pairs
examples = [
    {"query": q, "ground_truths": [eval_answers[i]]}
    for i, q in enumerate(eval_questions)
]

# create evaluation chains
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
# answer_relevancy_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_precision_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)

# Generate predictions
predictions = chain.batch(examples)
# predictions

faithfulness_scores = faithfulness_chain.evaluate(examples, predictions)
print(faithfulness_scores)
# answer_relevancy_scores = answer_relevancy_chain.evaluate(examples, predictions)
# print(answer_relevancy_scores)
context_precision_scores = context_precision_chain.evaluate(
    examples, predictions
)
print(context_precision_scores)
context_recall_scores = context_recall_chain.evaluate(examples, predictions)
print(context_recall_scores)
for i, score in enumerate(faithfulness_scores):
    predictions[i].update(score)
# for i, score in enumerate(answer_relevancy_scores):
#     predictions[i].update(score)
for i, score in enumerate(context_precision_scores):
    predictions[i].update(score)
for i, score in enumerate(context_recall_scores):
    predictions[i].update(score)

df_scores = pd.DataFrame(predictions)
df_scores

# # Display average scores
mean_faithfulness = df_scores["faithfulness_score"].mean()
# mean_answer_relevancy = df_scores['answer_relevancy_score'].mean()
mean_context_precision = df_scores["context_precision_score"].mean()
mean_context_recall = df_scores["context_recall_score"].mean()

print(f"mean_faithfulness: {mean_faithfulness}")
# print(f"mean_answer_relevancy: {mean_answer_relevancy}")
print(f"mean_context_precision: {mean_context_precision}")
print(f"mean_context_recall: {mean_context_recall}")
