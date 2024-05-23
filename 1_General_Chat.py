import streamlit as st
import os
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationChain

# webpage title setting
st.set_page_config(page_title="OpenGPT-General Chat", page_icon="📎")
st.header("Welcome to General Chat Window")

# LLM - Ollama(llama3)
# llm = ChatOllama(model="llama3")

# LLM - OpenAI
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API"))

# Memory
general_chat_msgs = StreamlitChatMessageHistory(key="general_chat_messages")
general_chat_memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="history",
    chat_memory=general_chat_msgs,
    return_messages=True,
)

chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=general_chat_memory,
)

btn_newchat = st.sidebar.button(label="New Chat")
# Initialize st_chat history and create message container
if len(general_chat_msgs.messages) == 0 or btn_newchat:
    general_chat_msgs.clear()
    general_chat_msgs.add_ai_message("How can I help you today?")

# Display history message
avatars = {"human": "user", "ai": "assistant"}
for msg in general_chat_msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# User/AI Conversation
if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        answer = chain.invoke(input=prompt)
        st.markdown(answer["response"])
