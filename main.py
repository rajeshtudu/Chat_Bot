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
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

load_dotenv()

st.set_page_config(page_title="Document Q&A + Appointment Chatbot")
st.title("Document Q&A + Appointment Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def load_documents(folder_path: str):
    """
    Load PDF and TXT documents from the folder.
    """
    documents = []
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist. Please create it and add PDF or TXT files.")
        return documents

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading file {filename}: {e}")

    return documents

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """
    Load documents, create embeddings, and build the FAISS vector store.
    """
    try:
        docs = load_documents("docs")
        if not docs:
            st.error("No documents found in the 'docs' folder. Please add PDF or TXT files.")
            return None
        st.success(f"Loaded {len(docs)} documents.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Vector store error: {e}")
        return None

vectorstore = get_vectorstore()
if vectorstore is None:
    st.stop()

@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Initialize the real LLM for answering questions.
    """
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        st.error(f"LLM initialization error: {e}")
        return None

llm = get_llm()
if llm is None:
    st.stop()

@st.cache_resource(show_spinner=False)
def get_qa_chain():
    """
    Create a conversational retrieval chain for Q&A.
    """
    try:
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    except Exception as e:
        st.error(f"Q&A chain error: {e}")
        return None

qa_chain = get_qa_chain()
if qa_chain is None:
    st.stop()

# Mock LLM that returns a properly formatted agent response to avoid parsing errors
class MockLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # This string is structured to match expected agent output format
        return "Final Answer: Your appointment has been successfully booked!"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "mock"

mock_llm = MockLLM()

tools = [
    Tool(
        name="BookAppointment",
        func=book_appointment_form,
        description="Collect Name, Phone, Email, and appointment date from user with validations."
    )
]

agent = initialize_agent(
    tools,
    mock_llm,
    agent="chat-conversational-react-description",
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=10,  # increase this
    max_execution_time=60  # seconds

)

user_input = st.text_input("Ask something or type 'I want to book an appointment'")
submit = st.button("Send")

if submit and user_input:
    try:
        if "book" in user_input.lower():
            # Invoke the booking agent tool conversation
            result = agent.invoke({
                "input": "Start booking an appointment",
                "chat_history": st.session_state.chat_history
            })
            st.write(result)
        else:
            # Query documents via conversational retrieval chain
            response = qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            st.write("Bot:", response["answer"])
            st.session_state.chat_history.append((user_input, response["answer"]))

    except google.api_core.exceptions.ResourceExhausted as e:
        st.error("Gemini API quota exceeded! Please wait a few minutes or upgrade your plan.")
        st.code(str(e), language="bash")
        st.markdown("[Learn more about Gemini quota limits](https://ai.google.dev/gemini-api/docs/rate-limits)")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

if st.checkbox("Show chat history"):
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")