# MahiBot - Personal RAG Chatbot

## Student Details
- **Name:** Maham Yousaf
- **Student ID:** F2024065505

## Project Description
MahiBot is a RAG (Retrieval-Augmented Generation) based chatbot designed to answer questions about my personal experiences. 
It uses LangChain, FAISS vector store, and Groq's Llama-3.3-70b-versatile model to provide accurate, context-aware responses based on my provided dataset (`experience_details.txt`).

## Live Deployment
You can access my live application here:
(https://mahi-bot-ed4jqmimeghax6brkwwu4j.streamlit.app/)

## How to Run Locally
1. Clone the repository: 
   `git clone https://github.com/Maham-yousaf13/Mahi-Bot.git`
2. Create a `.env` file in the folder and add your `GROK_API_KEY`:
   `GROK_API_KEY=your_actual_key_here`
3. Install dependencies: 
   `pip install -r requirements.txt`
4. Run the app: 
   `streamlit run app.py`
