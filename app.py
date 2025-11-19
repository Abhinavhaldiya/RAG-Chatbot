import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")



## Setup streamlit app

st.title('Conversational RAG with PDF uploads and chat history')
st.write("Upload PDF's and chat with their content")
st.set_page_config(
    page_title="Conversational RAG",
    page_icon="🤖",
)
llm=ChatGroq(model="openai/gpt-oss-20b",groq_api_key=groq_api_key)

## Chat Interface
session_id=st.text_input("Session ID",value="default_session")

if "store" not in st.session_state:
    st.session_state["store"] = {}


uploaded_files=st.file_uploader("Choose A Pdf file",type='pdf',accept_multiple_files=True)

## Process uploaded files
documents=[]
if uploaded_files:
    for upload_file in uploaded_files:
        file_name = upload_file.name
        temppdf = f"./temp_{file_name}"
        with open(temppdf,"wb") as file:
            file.write(upload_file.getvalue())
            file_name=upload_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)

# Debug
st.write("Docs loaded:", len(documents))

if len(documents) == 0:
    st.error("PDF contains no extractable text. Try another file or use OCR.")
    st.stop()

## Split and create Embedding for it
embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
splits=text_splitter.split_documents(documents)
vectorstore=FAISS.from_documents(documents=splits,embedding=embeddings)
retriever=vectorstore.as_retriever()

st.write("Splits created:", len(splits))


contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)
query_rewriter = contextualize_q_prompt | llm | StrOutputParser()

history_aware_retriever = (
    query_rewriter         
    | retriever             
)

system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
)

qa_prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])

question_answer_chain=qa_prompt|llm|StrOutputParser()|(lambda answer: {"answer": answer})

rag_chain=(
    {
    "context":history_aware_retriever,
    "input": lambda x: x["input"],
    "chat_history": lambda x: x["chat_history"]
    } | question_answer_chain
)

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
)

def format_chat_history(history):
    output = []
    for msg in history.messages:
        role = "User" if msg.type == "human" else "Assistant"
        output.append(f"{role}: {msg.content}\n")
    return "".join(output)

user_input=st.text_input("Your question:")
if user_input:
    session_history=get_session_history(session_id)
    response=conversational_rag_chain.invoke(
        {"input":user_input},
        config={
            "configurable":{"session_id":session_id}
        }
    )
    with st.sidebar:
        st.header("💬 Chat History")

        if len(session_history.messages) == 0:
            st.write("No messages yet.")
        else:
            for i, msg in enumerate(session_history.messages):
                if msg.type == "human":
                    st.markdown(
                        f"""
                        <div style="padding:10px; margin-bottom:10px; background:#1f2937; border-radius:8px;">
                            <strong>🧑 You:</strong><br>{msg.content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="padding:10px; margin-bottom:10px; background:#111827; border-radius:8px;">
                            <strong>🤖 Assistant:</strong><br>{msg.content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        history_text = format_chat_history(session_history)
        st.download_button(
            label="⬇️ Download Chat History",
            data=history_text,
            file_name=f"chat_history_{session_id}.txt",
            mime="text/plain",
            key="download_chat_history"
        ) 
    st.success(f"Assistant: {response['answer']}")

