import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.agents import initialize_agent, Tool
from utils.form_tools import book_appointment_form
from utils.validation import validate_email, validate_phone, parse_date
from dotenv import load_dotenv
import google.api_core.exceptions
import os
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load environment variables
load_dotenv()

st.set_page_config(page_title="📚 Document Q&A + Appointment Chatbot")
st.title("📚 Document Q&A + Appointment Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load documents
def load_documents(folder_path: str):
    documents = []
    if not os.path.exists(folder_path):
        st.error(f"❌ The folder '{folder_path}' does not exist. Please create it and add PDF or TXT files.")
        return documents
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
    return documents

# Retry logic for embedding
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def embed_documents(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    try:
        docs = load_documents("docs")
        if not docs:
            st.error("❌ No documents found in the 'docs' folder. Please add PDF or TXT files.")
            return None
        st.success(f"✅ Loaded {len(docs)} chunks.")
        
        # Optional: Limit to first few chunks to avoid API timeout
        limited_docs = docs[:10]  # limit to first 10 chunks
        st.info(f"🔍 Embedding {len(limited_docs)} chunks (limited for performance).")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        return embed_documents(limited_docs, embeddings)

    except Exception as e:
        st.error(f"❌ Vector store error: {e}")
        return None

vectorstore = get_vectorstore()
if vectorstore is None:
    st.stop()

@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        st.error(f"❌ LLM initialization error: {e}")
        return None

llm = get_llm()
if llm is None:
    st.stop()

@st.cache_resource(show_spinner=False)
def get_qa_chain():
    try:
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    except Exception as e:
        st.error(f"❌ Q&A chain error: {e}")
        return None

qa_chain = get_qa_chain()
if qa_chain is None:
    st.stop()

tools = [
    Tool(
        name="BookAppointment",
        func=book_appointment_form,
        description="Collect Name, Phone, Email, and appointment date from user with validations."
    )
]
agent = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=False)

user_input = st.text_input("💬 Ask something or type 'I want to book an appointment'")
submit = st.button("Send")

if submit and user_input:
    try:
        if "book" in user_input.lower():
            result = agent.invoke({
                "input": "Start booking an appointment",
                "chat_history": st.session_state.chat_history
            })
            st.write(result)
        else:
            response = qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            st.write("🤖:", response["answer"])
            st.session_state.chat_history.append((user_input, response["answer"]))

    except google.api_core.exceptions.ResourceExhausted as e:
        st.error("🚨 Gemini API quota exceeded! Please wait a few minutes or upgrade your plan.")
        st.code(str(e), language="bash")
        st.markdown("[🔗 Learn more about Gemini quota limits](https://ai.google.dev/gemini-api/docs/rate-limits)")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

if st.checkbox("🕓 Show chat history"):
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")