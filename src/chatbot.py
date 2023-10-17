import random
import time
from pathlib import Path

import numpy as np
import openai
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp


# Load environment variables
dotenv_path = Path("../.env")

if not load_dotenv(dotenv_path):
    print(
        "Could not load .env file or it is empty. Please check if it exists"
        " and is readable."
    )
    exit(1)


# Initialize token count
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

# LAYOUT
# st.set_page_config(layout='wide')
st.title("LLM Chatbot")

# Sidebar Layout
with st.sidebar:
    st.title("LLM Chatbot")
    st.write("Made by Keith Chan")
    st.divider()

    st.markdown("# About")
    st.write(
        """LLM Chatbot is just a UI for LLM testing with different models
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
                1. 💬 Ask questions about your document
            """
        )
    st.divider()

    # Display number of tokens used
    token_counter = st.empty()
    with token_counter:
        st.write(f"Total tokens used: {st.session_state.token_count}")
    st.write("Made by Keith Chan")

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
        temperature=0,
        n_gpu_layers=1,
    )
st.divider()

st.text_area(
    "question",
    key="question",
    height=100,
    placeholder="Enter question here",
    help="",
    label_visibility="collapsed",
)

if st.button("Get Answer", type="primary"):
    question = st.session_state.get("question", "")
    with st.spinner("preparing answer"):
        if selected_model == "GPT-3.5-turbo" or selected_model == "GPT-4":
            time_start = time.time()
            response = llm.predict(
                f"""
                {question}
            """
            )
            time_elapsed = time.time() - time_start
        else:
            time_start = time.time()
            # response = llm(f"""
            #     {question}
            # """)
            response = llm(
                f"""
                [INST]<<SYS>> You are a friendly, helpful assistant. Use three sentences maximum and keep the answer concise.<</SYS>>
                Question: {question}
                Answer: [/INST]
            """
            )
            time_elapsed = time.time() - time_start
    st.write(response)
    st.write(f"Response time: {time_elapsed:.2f} sec")
