import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.llms import LlamaCpp
from langchain.llms import CTransformers
from langchain.embeddings import LlamaCppEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
import time

# Initialize Session State Variables
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if 'token_count' not in st.session_state:
    st.session_state.token_count = 0

def click_submit():
    st.session_state.submitted = True

def add_spacer(num: int):
    for i in range(num):
        st.write("\n")


# Load environment variables
dotenv_path = Path('../.env')

if not load_dotenv(dotenv_path):
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# OpenAI config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Define the LLM
# llm = ChatOpenAI(temperature=0,model_name="gpt-4")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# llm = LlamaCpp(model_path="../models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=4048, temperature=0, max_tokens=0) #n_ctx is number of tokens used for context, and max_tokens is length of response #Source: https://swharden.com/blog/2023-07-29-ai-chat-locally-with-python/
# config = {'max_new_tokens': 512, 'temperature': 0.01}
# llm = CTransformers(model='../models/llama-2-7b-chat.ggmlv3.q8_0.bin',
#                     model_type='llama', config=config)

# LAYOUT
# st.set_page_config(page_title="DOCAI", page_icon="🤖", layout="wide", )     
st.title("Document Q&A with RAG")

# Sidebar Layout
with st.sidebar:
    st.title("Document Q&A with RAG")
    st.write("Made by Keith Chan")
    st.markdown("This is a Proof of Concept system and may contain bugs or unfinished features.")
    st.markdown("Source code can be found [here].")
    with st.expander('What is this app about?'):
        st.markdown("""
            This app is designed to implement Retrieval Augmented Generation using locall-run LLMs.\n
            Instructions:
                - User uploads PDF document (later versions will allow other file types)
                - Provide queries to the document. 
            """
        )
    placeholder = st.empty()
    with placeholder.container():
        add_spacer(15)

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
    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()

    # Vector Store
    db = Chroma.from_documents(documents=texts, embedding=embeddings)
    return db

uploaded_file = st.file_uploader("Upload a PDF File (No bigger than 200mb)", type="pdf", accept_multiple_files=False)

_, _, _, _, _, col6 = st.columns(6) # To right-align submit button
submit_button = col6.button('Submit', on_click=click_submit)

# Data Ingestion
if uploaded_file is not None and st.session_state.submitted:
    bytes_data = uploaded_file.getvalue()

    dir_path = "../temp/"
    filename = uploaded_file.name
    save_path = Path(dir_path, filename)
    with open(save_path, 'wb') as f: 
        f.write(bytes_data)
        
    if save_path.exists():
        st.success(f'File {filename} is successfully saved!')

    db = create_embeddings_and_vectorstore(save_path)

    # Create Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    {context}

    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate.from_template(template)

    # Initialise RetrievalQA Chain
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_kwargs={"k":2}),#search_type="mmr"),#search_kwargs={"k":3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    st.success(f'chain created!')


# React to user input
prompt = st.chat_input("Say something")
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner('Preparing answer...'):

        with get_openai_callback() as cb: # Token usage only possible for OpenAI API
            time_start = time.time()
            response = chain({"query": prompt})
            time_elapsed = time.time() - time_start

        # Display assistant response in chat message container
        # with st.chat_message("assistant"):

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## Answer")
            st.markdown(response['result'])
            st.write(f'Response time: {time_elapsed:.2f} sec')
        with col2:
            # Provide source documents
            st.markdown('## Sources')
            with st.expander('Source Documents'):
                for i, source_doc in enumerate(response['source_documents']):
                    st.markdown(f'### Source Document {i+1}')
                    st.write(source_doc.page_content)
                    st.write(f'Page {source_doc.metadata["page"]}')
                    st.write(f'Source file: {source_doc.metadata["source"]}')
                    st.divider()

    # Increment tokens used
    with token_counter: #TODO: accumulate tokens used. Currently shows each call
        st.write(cb)