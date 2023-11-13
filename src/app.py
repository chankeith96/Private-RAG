import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

# Initialize Session State Variables
if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "token_count" not in st.session_state:
    st.session_state.token_count = 0


def click_submit():
    st.session_state.submitted = True


def add_spacer(num: int):
    for i in range(num):
        st.write("\n")


# Load environment variables
dotenv_path = Path("../.env")

if not load_dotenv(dotenv_path):
    print(
        "Could not load .env file or it is empty. Please check if it exists"
        " and is readable."
    )
    exit(1)

# OpenAI config
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY"
)  # TODO: consider removing, currently not used

# LAYOUT
# st.set_page_config(page_title="DOCAI", page_icon="🤖", layout="wide", )
st.title("Document Q&A with RAG")

# Sidebar Layout
with st.sidebar:
    st.title("🦙📄💬 Document Q&A with RAG")
    st.write("Made by Keith Chan")
    st.divider()

    st.markdown("# About")
    st.write(
        """Document Q&A allows you to ask questions about your
        documents and get accurate answers with instant citations.
        This app is designed to implement Retrieval Augmented Generation
        using locally-hosted LLMs for complete data privacy and security.
        """
    )
    st.warning(
        "This is a Proof of Concept system and may contain bugs or unfinished"
        " features."
    )
    st.markdown(
        "Source code can be found [here](https://github.com/chank20/RAG-Implementation)."
    )
    st.divider()

    with st.expander("How to use"):
        st.markdown(
            """
                1. 📄 Upload a single PDF document (later versions will allow other file types and multiple documents)
                2. 💬 Ask questions about your document
            """
        )
    st.divider()

    placeholder = st.empty()
    with placeholder.container():
        add_spacer(1)

    # Display number of tokens used
    token_counter = st.empty()
    with token_counter:
        st.write(f"Total tokens used: {st.session_state.token_count}")
    st.write("Made by Keith Chan")


@st.cache_resource(ttl=1800)
def create_embeddings_and_vectorstore(file_path):
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    # Chunk and Embeddings
    # text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""],  # adjust these as necessary
    )

    texts = text_splitter.split_documents(pages)

    # embeddings = OpenAIEmbeddings()

    # embeddings = LlamaCppEmbeddings(
    #     model_path="../models/llama-2-7b-chat.Q4_K_M.gguf",
    #     n_gpu_layers=1,
    #     n_batch=512,
    #     n_ctx=2000,
    # )

    # embeddings = LlamaCppEmbeddings(model_path="../models/llama-2-13b-chat.Q5_K_M.gguf", n_gpu_layers=1, n_batch=512, n_ctx=2000)

    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name
    # )  # , model_kwargs={"device": "cuda"})

    # emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=emb_model,
    #     # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME') # TODO Look into cache_folder param
    # )

    emb_model = "hkunlp/instructor-large"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=emb_model,
        # cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME') # TODO Look into cache_folder param
    )

    # Vector Store
    db = Chroma.from_documents(documents=texts, embedding=embeddings)
    return db


# Define the LLM
model_list = [
    "GPT-3.5-turbo",
    "GPT-4",
    "Llama2-7B (4bit)",
    "Llama2-7B (8bit)",
    "Llama2-13B (5bit)",
]
selected_model = st.selectbox(
    "Choose a model", options=model_list, key="selected_model"
)
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
st.divider()

uploaded_file = st.file_uploader(
    "Upload a PDF File (No bigger than 200mb)",
    type="pdf",
    accept_multiple_files=False,
)

_, _, _, _, _, col6 = st.columns(6)  # To right-align submit button
submit_button = col6.button("Submit", on_click=click_submit)

# Data Ingestion
if uploaded_file is not None and st.session_state.submitted:
    bytes_data = uploaded_file.getvalue()

    dir_path = "../temp/"
    filename = uploaded_file.name
    save_path = Path(dir_path, filename)
    with open(save_path, "wb") as f:
        f.write(bytes_data)

    if save_path.exists():
        st.success(f"File {filename} is successfully saved!")

    time_start = time.time()
    db = create_embeddings_and_vectorstore(save_path)
    time_elapsed = time.time() - time_start

    st.write(f"Embedding time: {time_elapsed:.2f} sec")
    # Create Prompt
    # template = """Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    # {context}

    # Question: {question}
    # Answer:
    # """

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
        conditionals=[
            (lambda llm: isinstance(llm, LlamaCpp), llama_rag_prompt)
        ],
    )

    # TODO: Can be removed. Only used to debug prompt selection
    system_prompt = rag_prompt_selector.get_prompt(llm)
    st.write(system_prompt.template)

    # Initialise RetrievalQA Chain
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(
            search_kwargs={"k": 2}
        ),  # search_type="mmr"),#search_kwargs={"k":3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt_selector.get_prompt(llm)},
    )
    st.success("chain created!")


# React to user input
prompt = st.chat_input("Say something")
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Preparing answer..."):
        with get_openai_callback() as cb:  # Token usage only possible for OpenAI API
            time_start = time.time()
            response = chain({"query": prompt})
            time_elapsed = time.time() - time_start

        # Display assistant response in chat message container
        # with st.chat_message("assistant"):

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## Answer")
            st.markdown(response["result"])
            st.write(f"Response time: {time_elapsed:.2f} sec")
        with col2:
            # Provide source documents
            st.markdown("## Sources")
            with st.expander("Source Documents"):
                for i, source_doc in enumerate(response["source_documents"]):
                    st.markdown(f"### Source Document {i+1}")
                    st.write(source_doc.page_content)
                    st.write(f'Page {source_doc.metadata["page"]}')
                    st.write(f'Source file: {source_doc.metadata["source"]}')
                    st.divider()

    # Increment tokens used
    with (
        token_counter
    ):  # TODO: accumulate tokens used. Currently shows each call
        st.write(cb)
