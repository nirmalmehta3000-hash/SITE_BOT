from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

import streamlit as st
import pandas as pd
from datetime import datetime
import re
import uuid

# LangChain imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Import database utilities
try:
    from db_utils import (
        initialize_database,
        save_chat_to_database,
        get_user_chat_history
    )
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"Database utilities not found: {e}")

# API Keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

VECTOR_DB_PATH = "vectorstore.faiss"
DATASET_PATH = "dataset.xlsx"

# System instruction
SYSTEM_INSTRUCTION = (
    "You are a friendly, insightful customer service assistant for www.gerrysonmehta.com. "
    "Your expertise is helping aspiring data analysts, students, and professionals with career advice, "
    "project ideas, interview prep, portfolio building and time management. "
    "Respond conversationally, with empathy, encouragement and actionable steps; like a human expert mentor. "
    "Always personalize your guidance, use clear language and be honest when uncertain."
)

# Prompt template
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user's question.
Respond conversationally, as a human expert coach.
If you don't know the answer, just say so honestly and avoid guessing.
----------------
{context}
Question: {question}
Expert Answer:
"""

# ============================================
# UTILITY FUNCTIONS
# ============================================
def validate_mobile_number(mobile):
    """Validate mobile number"""
    if not mobile:
        return False, "Mobile number is required"
    
    clean_mobile = re.sub(r'[\s\-\+\(\)]', '', mobile)
    
    if not clean_mobile.isdigit():
        return False, "Mobile number should contain only digits"
    
    if len(clean_mobile) < 7 or len(clean_mobile) > 15:
        return False, "Mobile number should be between 7-15 digits"
    
    return True, "Valid"

def validate_email(email):
    """Validate email format"""
    if not email:
        return False, "Email is required"
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Please enter a valid email address"
    
    return True, "Valid"

def validate_name(name):
    """Validate name"""
    if not name or len(name.strip()) < 2:
        return False, "Name must be at least 2 characters long"
    
    return True, "Valid"

# ============================================
# VECTORSTORE / QA
# ============================================
def create_vector_db():
    """Create vector database from Excel dataset"""
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found. Please upload {DATASET_PATH} to your project.")
        st.info("Create a dataset.xlsx file with 'prompt' and 'response' columns.")
        return None
        
    try:
        df = pd.read_excel(DATASET_PATH)
        
        if 'prompt' not in df.columns or 'response' not in df.columns:
            st.error("Dataset must have 'prompt' and 'response' columns.")
            return None
            
        df = df.dropna(subset=['prompt', 'response'])
        
        if len(df) == 0:
            st.error("No valid data found in dataset.")
            return None
            
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None

    try:
        documents = [
            Document(page_content=f"Q: {row.get('prompt','')}\nA: {row.get('response','')}")
            for _, row in df.iterrows()
        ]
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        
        st.success(f"Knowledge base created with {len(texts)} chunks!")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

def get_vectorstore():
    """Get or create vectorstore"""
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if not os.path.exists(VECTOR_DB_PATH):
            st.warning("Creating knowledge base from dataset...")
            vectorstore = create_vector_db()
            return vectorstore
            
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def get_qa_chain():
    """Create QA chain"""
    if not GOOGLE_API_KEY:
        st.error("Google API Key not found. Please set GOOGLE_API_KEY in your environment.")
        return None
        
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None
        
    try:
        llm = GoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
        
        return RetrievalQA(
            retriever=retriever,
            combine_documents_chain=stuff_chain,
            return_source_documents=False
        )
        
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# ============================================
# USER INFORMATION COLLECTION
# ============================================
def collect_user_info():
    """Collect user information step by step"""
    st.title("🤖 Welcome to Gerryson Mehta's AI Assistant")
    st.markdown("*Your Career Growth Partner*")
    
    # Initialize step if not exists
    if 'info_step' not in st.session_state:
        st.session_state.info_step = 1
        st.session_state.temp_user_data = {}
    
    # Step 1: Name
    if st.session_state.info_step == 1:
        st.subheader("👋 Let's get started!")
        st.write("I'd love to personalize our conversation. What's your name?")
        
        name = st.text_input(
            "Your Full Name",
            placeholder="Enter your full name",
            key="name_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Next →", disabled=not name):
                name_valid, name_msg = validate_name(name)
                if name_valid:
                    st.session_state.temp_user_data['name'] = name.strip()
                    st.session_state.info_step = 2
                    st.rerun()
                else:
                    st.error(name_msg)
    
    # Step 2: Email
    elif st.session_state.info_step == 2:
        st.subheader(f"Nice to meet you, {st.session_state.temp_user_data['name']}! 👋")
        st.write("What's your email address?")
        
        email = st.text_input(
            "Email Address",
            placeholder="your.email@example.com",
            key="email_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("← Back"):
                st.session_state.info_step = 1
                st.rerun()
        
        with col2:
            if st.button("Next →", disabled=not email):
                email_valid, email_msg = validate_email(email)
                if email_valid:
                    st.session_state.temp_user_data['email'] = email.strip().lower()
                    st.session_state.info_step = 3
                    st.rerun()
                else:
                    st.error(email_msg)
    
    # Step 3: Mobile
    elif st.session_state.info_step == 3:
        st.subheader("Almost there! 📱")
        st.write("What's your mobile number?")
        
        mobile = st.text_input(
            "Mobile Number",
            placeholder="+1234567890 or 1234567890",
            key="mobile_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("← Back"):
                st.session_state.info_step = 2
                st.rerun()
        
        with col2:
            if st.button("Start Chat! 🚀", disabled=not mobile):
                mobile_valid, mobile_msg = validate_mobile_number(mobile)
                if mobile_valid:
                    st.session_state.temp_user_data['mobile'] = mobile.strip()
                    
                    # Save user data to session
                    st.session_state.user_data = {
                        'name': st.session_state.temp_user_data['name'],
                        'email': st.session_state.temp_user_data['email'],
                        'mobile': st.session_state.temp_user_data['mobile'],
                        'user_id': str(uuid.uuid4())
                    }
                    
                    st.session_state.user_info_collected = True
                    st.session_state.chat_history = []
                    
                    # Clean up temporary data
                    del st.session_state.temp_user_data
                    del st.session_state.info_step
                    
                    st.success(f"Perfect! Welcome aboard, {st.session_state.user_data['name']}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(mobile_msg)

# ============================================
# CHAT INTERFACE
# ============================================
def chat_interface():
    """Main chat interface"""
    user_data = st.session_state.user_data
    
    st.title("🤖 Gerryson Mehta's AI Assistant")
    st.markdown(f"*Welcome back, {user_data['name']}! How can I help you today?*")
    
    # Initialize QA chain
    if 'qa_chain' not in st.session_state:
        with st.spinner("Loading AI assistant..."):
            st.session_state.qa_chain = get_qa_chain()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your career, projects, or technical questions..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate response
        if st.session_state.qa_chain:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = response.get("result", "I couldn't generate a proper response.")
                except Exception as e:
                    answer = f"Sorry, I encountered an error: {str(e)}"
        else:
            answer = "AI assistant is not available. Please check your configuration."
        
        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Save to database
        if DB_AVAILABLE:
            try:
                save_chat_to_database(
                    user_data['name'],
                    user_data['email'], 
                    user_data['mobile'],
                    prompt,
                    answer
                )
            except Exception as e:
                st.error(f"Failed to save chat: {e}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Add a reset button
    if st.button("🔄 Start New Conversation"):
        # Keep user data but clear chat
        chat_history = st.session_state.chat_history
        st.session_state.clear()
        st.session_state.user_data = user_data
        st.session_state.user_info_collected = True
        st.session_state.chat_history = []
        st.rerun()

# ============================================
# MAIN APP
# ============================================
def main():
    """Main application"""
    st.set_page_config(
        page_title="Gerryson Mehta AI Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Hide sidebar completely
    st.markdown("""
        <style>
        .css-1d391kg {display: none;}
        .css-1rs6os {display: none;}
        .css-17ziqus {display: none;}
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize database
    if DB_AVAILABLE:
        initialize_database()
    
    # Initialize session state
    if 'user_info_collected' not in st.session_state:
        st.session_state.user_info_collected = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Route to appropriate interface
    if not st.session_state.user_info_collected:
        collect_user_info()
    else:
        chat_interface()

if __name__ == "__main__":
    main()
