import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.agents import initialize_agent, Tool
from utils.form_tools import book_appointment_form
from utils.validation import validate_email, validate_phone, parse_date
from dotenv import load_dotenv
import os

# Hugging Face transformers for DialoGPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load env variables
load_dotenv()

st.set_page_config(page_title="📚 Document Q&A + Appointment Chatbot with DialoGPT")
st.title("📚 Document Q&A + Appointment Chatbot with DialoGPT")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load documents from 'docs' folder
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
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.error(f"❌ Failed to load {filename}: {e}")
    return documents

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    try:
        docs = load_documents("docs")
        if not docs:
            st.error("❌ No documents found in the 'docs' folder. Please add PDF or TXT files.")
            return None
        st.success(f"✅ Loaded {len(docs)} documents.")
        # Use HuggingFace embeddings for vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"❌ Vector store error: {e}")
        return None

vectorstore = get_vectorstore()
if vectorstore is None:
    st.stop()

@st.cache_resource(show_spinner=False)
def get_qa_chain():
    try:
        # Use a dummy llm placeholder because we'll use DialoGPT for conversation
        # But LangChain needs an llm for ConversationalRetrievalChain; we use a dummy here
        from langchain.llms.base import LLM
        class DummyLLM(LLM):
            def _call(self, prompt, stop=None):
                return "Answering..."
            @property
            def _identifying_params(self):
                return {}
            @property
            def _llm_type(self):
                return "dummy"
        dummy_llm = DummyLLM()
        return ConversationalRetrievalChain.from_llm(llm=dummy_llm, retriever=vectorstore.as_retriever())
    except Exception as e:
        st.error(f"❌ Q&A chain error: {e}")
        return None

qa_chain = get_qa_chain()
if qa_chain is None:
    st.stop()

# Initialize DialoGPT from HuggingFace for conversation
@st.cache_resource(show_spinner=False)
def load_dialo_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, dialo_model = load_dialo_model()

# Appointment tool and agent (same as before)
tools = [
    Tool(
        name="BookAppointment",
        func=book_appointment_form,
        description="Collect Name, Phone, Email, and appointment date from user with validations."
    )
]
# Initialize agent with dummy LLM, since main chat uses DialoGPT
agent = initialize_agent(tools, llm=None, agent="chat-conversational-react-description", verbose=False)

# Store dialoGPT chat history tokens
if "dialo_history_ids" not in st.session_state:
    st.session_state.dialo_history_ids = None

user_input = st.text_input("💬 Ask something or type 'I want to book an appointment'")
submit = st.button("Send")

if submit and user_input:
    try:
        if "book" in user_input.lower():
            # Launch booking form
            result = agent.invoke({
                "input": "Start booking an appointment",
                "chat_history": st.session_state.chat_history
            })
            st.write(result)

        else:
            # First try to answer from documents with retrieval chain
            response = qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            answer = response.get("answer", "").strip()
            
            # If retrieval answer is empty or too short, fallback to DialoGPT conversational response
            if not answer or len(answer) < 5:
                # Prepare DialoGPT input tokens
                new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
                # Append chat history tokens
                if st.session_state.dialo_history_ids is not None:
                    bot_input_ids = torch.cat([st.session_state.dialo_history_ids, new_input_ids], dim=-1)
                else:
                    bot_input_ids = new_input_ids
                # Generate reply
                chat_history_ids = dialo_model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
                # Decode
                answer = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
                # Update chat history ids
                st.session_state.dialo_history_ids = chat_history_ids
                
            st.write("🤖:", answer)
            st.session_state.chat_history.append((user_input, answer))

    except Exception as e:
        st.error(f"Unexpected error: {e}")

if st.checkbox("🕓 Show chat history"):
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")