import streamlit as st
import os
import tempfile
import random
import string

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever

# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def get_file_type(filename):
    _, file_type = os.path.splitext(filename)
    return file_type.lower()


def file_load(uploaded_files):
    """read document"""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    loader_map = {
        ".doc": Docx2txtLoader,
        ".docx": Docx2txtLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }

    for file in uploaded_files:
        file_type = get_file_type(file.name)
        if file_type in loader_map:
            temp_path = os.path.join(temp_dir.name, file.name)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file.getvalue())
            loader = loader_map[file_type]
            docs.extend(loader(temp_path).load())

    for doc in docs:
        filename = os.path.splitext(os.path.basename(doc.metadata["source"]))[0]
        doc.metadata["filename"] = filename

    return docs


def file_splitter(docs):
    """document split"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    return splits


def generate_random_code(length):
    characters = string.ascii_letters + string.digits
    filder_id = "".join(random.choice(characters) for _ in range(length))
    return filder_id


def embedding_to_vector(document_splits, docstore_id=None):

    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model = HuggingFaceEmbeddings(model_name=model_name)

    persist_directory = f"./Chroma/{docstore_id}"
    vectorstore = Chroma.from_documents(
        documents=document_splits,
        embedding=model,
        persist_directory=persist_directory,
        collection_name="opengpt",
    )

    document_content_description = "These are the information about the conversation."

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source location of the document.",
            type="string",
        ),
        AttributeInfo(
            name="filename", description="The name of the document", type="string"
        ),
    ]

    # LLM - Ollama(llama3)
    # llm = ChatOllama(model="llama3")

    # LLM - OpenAI
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API"))

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 8},
    )

    return retriever


# add filename into filtered_docs page_content
def doc_content_add_filename(docs):
    for doc in docs:
        doc.page_content = (
            "File Name:" + doc.metadata["filename"] + "\n" + doc.page_content
        )
    return docs


# answer source extract
def source_extract(ai_response):
    sources = []
    for source in ai_response["source_documents"]:
        sources.append(os.path.split(source.metadata["source"])[1])
    unique_sources = list(set(sources))

    return unique_sources


# streamlit button status for button upload
def upload_initial_status_update():
    st.session_state.upload_initial = False


# Memory initial
document_chat_msgs = StreamlitChatMessageHistory(key="document_chat_messages")

# webpage title setting
st.set_page_config(page_title="OpenGPT-Document Chat", page_icon="📎")
st.title("Chat with Your Documents")


# 新對話視窗(新記憶位置)
if st.sidebar.button(label="New Chat", type="secondary"):
    document_chat_msgs.clear()
    del st.session_state.file_uploader
    del st.session_state.upload_initial
    del st.session_state.doc_store_id
    del st.session_state.retriever


if "upload_initial" not in st.session_state:
    st.session_state.upload_initial = True
    st.session_state.doc_store_id = generate_random_code(16)
    st.session_state.retriever = []

# Files 在觸發 form_submit_button 才會改變
with st.sidebar.form(key="form_fileloader", clear_on_submit=True):
    files = st.file_uploader(
        label="File Loader",
        type=["docx", "pptx", "csv", "pdf", "xlsx"],
        accept_multiple_files=True,
        label_visibility="hidden",
        key="file_uploader",
    )
    form_btn_upload = st.form_submit_button(
        "Upload", type="primary", on_click=upload_initial_status_update
    )

if st.session_state.upload_initial:
    st.info("Please upload your documents to continue.")
    st.stop()

elif len(files) > 0 and form_btn_upload:
    docs = file_load(files)
    splits = file_splitter(docs)
    st.session_state.retriever = embedding_to_vector(
        document_splits=splits, docstore_id=st.session_state.doc_store_id
    )
    st.info("Documents have already uploaded.")

elif not files and form_btn_upload:
    st.warning("Oops, there are no documents.")

# ------------------------------
# LLM - Ollama
# llm = ChatOllama(model="llama3")

# LLM - OpenAI
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API"))

# Memory
memory = ConversationBufferWindowMemory(
    return_messages=True,
    input_key="query",
    memory_key="chat_history",
    chat_memory=document_chat_msgs,
)

prompt_template = """
Use the following pieces of context and chat history to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Chat history:{chat_history}

Question: {query}
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "query", "context"], template=prompt_template
)

# Chain
chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
    memory=memory,
    prompt=prompt,
    verbose=True,
)


# Initialize st_chat history and create message container
if len(document_chat_msgs.messages) == 0:
    document_chat_msgs.add_ai_message("How can I help you today?")

# Display history message
avatars = {"human": "user", "ai": "assistant"}
for msg in document_chat_msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# User/AI Conversation
if query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        retriever = st.session_state.retriever
        filtered_docs = retriever.invoke(query)
        input_docs = doc_content_add_filename(filtered_docs)
        response = chain({"query": query, "input_documents": input_docs})
        st.markdown(response["output_text"])
