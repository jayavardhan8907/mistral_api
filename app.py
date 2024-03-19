import os
import time
import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pypdf import PdfReader

# Set page configuration
st.set_page_config(page_title="YOJANA SAATHI", page_icon="flag.jpeg", layout="wide")

# Initialize session state attributes
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "text" not in st.session_state:
    st.session_state.text = ""

# Add logo, title, and chatbot avatar
st.title("YOJANA SAATHI")
st.image("flag.jpeg", width=50)  # Adjust width as needed

def add_message(msg, agent="ai", stream=True, store=True):
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent, avatar="logo2.png"):  # Set chatbot avatar
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output))

@st.cache_resource
def get_client():
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)

CLIENT: MistralClient = get_client()

PROMPT = """
An excerpt from a document is given below.

---------------------
{context}
---------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer.
Do not output anything that can't be answered from the context.

Query: {query}
Answer:
"""

def reply(query: str, index: IndexFlatL2):
    embedding = embed(query)
    embedding = np.array([embedding])

    _, indexes = index.search(embedding, k=2)
    context = [st.session_state.chunks[i] for i in indexes.tolist()[0]]

    messages = [
        ChatMessage(role="user", content=PROMPT.format(context=context, query=query))
    ]
    response = CLIENT.chat_stream(model="mistral-tiny", messages=messages)

    add_message(stream_response(response))

def build_index(pdf_file_path):
    st.session_state.messages = []

    if not pdf_file_path:
        st.session_state.clear()
        return

    reader = PdfReader(pdf_file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n\n"

    st.session_state.text += text  # Append text to existing text

    chunk_size = 2048  # Increased chunk size
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    if len(chunks) > 500:  # Increased maximum number of chunks
        st.error("Document is too long!")
        st.session_state.clear()
        return

    st.session_state.chunks.extend(chunks)
    if st.session_state.index is None:
        dimension = len(embed(chunks[0]))
        st.session_state.index = IndexFlatL2(dimension)
    st.session_state.index.add(np.array([embed(chunk) for chunk in chunks]))

def stream_str(s, speed=250):
    for c in s:
        yield c
        time.sleep(1 / speed)

def stream_response(response):
    for r in response:
        yield r.choices[0].delta.content

@st.cache_data
def embed(text: str):
    return CLIENT.embeddings("mistral-embed", text).data[0].embedding

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["agent"], avatar="logo2.png"):  # Set chatbot avatar
        st.write(message["content"])

# Directly access files from a specific path
pdf_files_path = "database"
pdf_files = [os.path.join(pdf_files_path, f) for f in os.listdir(pdf_files_path) if f.endswith('.pdf')]

if not pdf_files:
    st.stop()

query = st.text_input("Ask something about your PDF")

if query:
    if not st.session_state.messages:
        for pdf_file_path in pdf_files:
            build_index(pdf_file_path)

        index: IndexFlatL2 = st.session_state.index
        reply(query, index)
        add_message("Ready to answer your questions.")

    add_message(query, agent="human", stream=False, store=True)
    reply(query, index)
