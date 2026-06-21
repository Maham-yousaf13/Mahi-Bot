import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Configure the page title and icon
st.set_page_config(page_title="MahiBot", page_icon="🤖")
st.title("🤖 MahiBot Assistant")
st.subheader("Your Personal Experience Expert")

# Load environment variables (API Key) [cite: 33, 34]
load_dotenv()

# Setup RAG pipeline and cache it to improve performance [cite: 33, 35]
@st.cache_resource
def setup_rag():
    # Load the custom dataset [cite: 23, 24]
    loader = TextLoader("experience_details.txt")
    documents = loader.load()
    
    # Split documents into smaller chunks for better retrieval [cite: 32, 35]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Generate embeddings and store in a vector database [cite: 36, 37]
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Initialize RAG and LLM [cite: 33, 39]
vectorstore = setup_rag()
# Initialize RAG and LLM with a currently supported model [cite: 39]
llm = ChatGroq(api_key=os.getenv("GROK_API_KEY"), model="llama-3.3-70b-versatile")
# Initialize chat history to maintain conversation context [cite: 41]
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input via chat interface 
if prompt := st.chat_input("Ask about my experience..."):
    # Display and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform retrieval and generate response using the LLM [cite: 38, 43]
    with st.chat_message("assistant"):
        retriever = vectorstore.as_retriever()
        # Using .invoke() instead of .get_relevant_documents() to avoid deprecation errors
        docs = retriever.invoke(prompt)
        context = "\n".join([d.page_content for d in docs])
        
        # Proper prompt engineering for context-aware answers [cite: 40]
        response = llm.invoke(f"Context: {context}\n\nQuestion: {prompt}")
        full_response = response.content
        st.markdown(full_response)
    
    # Save assistant response to history [cite: 41]
    st.session_state.messages.append({"role": "assistant", "content": full_response})