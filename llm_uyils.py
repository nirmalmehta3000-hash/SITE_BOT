import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from vector_utils import get_vectorstore

try:
    from langchain_xai import ChatXAI
except ImportError:
    ChatXAI = None

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROK_API_KEY = os.environ.get("GROK_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

PROMPT_TEMPLATE = """
Use the following context to answer the user's question.
If you don't know, say so honestly.
----------------
{context}
Question: {question}
Answer:
"""

def determine_llm(question: str):
    grok_keywords = ["grok", "xai", "x-ai", "grok model"]
    openrouter_keywords = ["claude", "gpt-4", "llama", "openrouter"]
    
    if any(word in question.lower() for word in grok_keywords) and ChatXAI and GROK_API_KEY:
        return "grok"
    elif any(word in question.lower() for word in openrouter_keywords) and OPENROUTER_API_KEY:
        return "openrouter"
    return "gemini"

def init_llm(choice: str):
    try:
        if choice == "gemini":
            return GoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY,
            )
        elif choice == "grok" and ChatXAI and GROK_API_KEY:
            return ChatXAI(
                model="grok-4",
                temperature=0.3,
                xai_api_key=GROK_API_KEY,
            )
        elif choice == "openrouter" and OPENROUTER_API_KEY:
            return ChatOpenAI(
                model="anthropic/claude-3.5-sonnet",
                temperature=0.3,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://gerrysonmehta.com",
                    "X-Title": "Gerryson Mehta AI Assistant",
                },
            )
    except Exception as e:
        st.error(f"Error initializing {choice}: {e}")
    return None

def get_qa_chain(llm):
    vectorstore = get_vectorstore()
    if not vectorstore:
        return None
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    return RetrievalQA(retriever=retriever, combine_documents_chain=stuff_chain, return_source_documents=False)
