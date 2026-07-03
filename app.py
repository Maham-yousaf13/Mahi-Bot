import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MahiBot - Professional AI Assistant", 
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS WITH GLASSY TEAL/MINT PALETTE
# ============================================================
st.markdown("""
<style>
    /* ─── Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ─── TEAL/MINT PALETTE ─── */
    /* 
        #003C43 = Deep Teal (Darkest)
        #135D66 = Medium Teal
        #77B0AA = Soft Teal/Mint
        #E3FEF7 = Light Mint (Lightest)
    */
    
    /* ─── Global ─── */
    .stApp {
        background: linear-gradient(135deg, #003C43 0%, #135D66 40%, #77B0AA 70%, #E3FEF7 100%) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Force white text for readability on dark gradient */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown strong, .stMarkdown em, .stMarkdown span, .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* ─── GLASS SIDEBAR ─── */
    .css-1d391kg, .css-1wrcr25 {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(30px) !important;
        -webkit-backdrop-filter: blur(30px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 40px rgba(0, 60, 67, 0.2) !important;
    }
    
    /* Sidebar Text */
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stMarkdown p,
    .css-1d391kg .stMarkdown div,
    .css-1d391kg .stMarkdown span,
    .css-1d391kg .stMarkdown strong,
    .css-1wrcr25 .stMarkdown,
    .css-1wrcr25 .stMarkdown p,
    .css-1wrcr25 .stMarkdown div,
    .css-1wrcr25 .stMarkdown span,
    .css-1wrcr25 .stMarkdown strong {
        color: #ffffff !important;
    }
    
    /* ─── GLASS SIDEBAR SECTIONS ─── */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        padding: 1.2rem 1.4rem !important;
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px rgba(0, 60, 67, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .sidebar-section:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 40px rgba(0, 60, 67, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    
    .sidebar-section h4 {
        font-size: 0.6rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        color: #E3FEF7 !important;  /* Light Mint */
        margin-bottom: 0.8rem !important;
        opacity: 0.9 !important;
    }
    
    .sidebar-section .label {
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    .sidebar-section .value {
        font-size: 0.82rem !important;
        color: #77B0AA !important;  /* Soft Teal/Mint */
        font-weight: 500 !important;
    }
    
    .contact-item {
        display: flex !important;
        align-items: center !important;
        gap: 0.8rem !important;
        font-size: 0.8rem !important;
        color: #ffffff !important;
        padding: 0.4rem 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
        transition: all 0.2s ease !important;
    }
    
    .contact-item:hover {
        padding-left: 0.5rem !important;
        border-bottom-color: rgba(119, 176, 170, 0.3) !important;
    }
    
    .contact-item:last-child {
        border-bottom: none !important;
    }
    
    .contact-item .icon {
        width: 22px !important;
        color: #77B0AA !important;  /* Soft Teal/Mint */
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .contact-item span {
        color: #ffffff !important;
    }
    
    /* ─── SIDEBAR DIVIDER ─── */
    hr {
        border: none !important;
        border-top: 2px solid rgba(255, 255, 255, 0.08) !important;
        margin: 1.2rem 0 !important;
        border-radius: 2px !important;
    }
    
    /* ─── MAIN HEADER ─── */
    .main-header {
        background: rgba(0, 60, 67, 0.5) !important;  /* Deep Teal with opacity */
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        padding: 2.5rem 3rem !important;
        border-radius: 20px !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 20px 60px -20px rgba(0, 60, 67, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .main-header .logo-icon {
        font-size: 2.2rem !important;
        display: block !important;
        color: #E3FEF7 !important;  /* Light Mint */
    }
    
    .main-header h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
        text-shadow: 0 2px 20px rgba(0,0,0,0.15) !important;
    }
    
    .main-header .subtitle {
        font-size: 0.95rem !important;
        font-weight: 300 !important;
        color: #77B0AA !important;  /* Soft Teal/Mint */
        margin-top: -0.2rem !important;
        opacity: 0.9 !important;
    }
    
    .main-header .divider-line {
        width: 50px !important;
        height: 2px !important;
        background: rgba(119, 176, 170, 0.5) !important;
        margin: 0.6rem auto !important;
        border-radius: 2px !important;
    }
    
    .badge-container {
        display: flex !important;
        justify-content: center !important;
        gap: 0.6rem !important;
        flex-wrap: wrap !important;
        margin-top: 0.6rem !important;
    }
    
    .badge {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(10px) !important;
        padding: 0.35rem 1.4rem !important;
        border-radius: 100px !important;
        font-size: 0.65rem !important;
        font-weight: 500 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        letter-spacing: 0.6px !important;
        text-transform: uppercase !important;
    }
    
    .badge.highlight {
        background: rgba(119, 176, 170, 0.15) !important;  /* Soft Teal with opacity */
        color: #ffffff !important;
        border-color: rgba(119, 176, 170, 0.2) !important;
    }
    
    /* ─── GLASS CHAT MESSAGES ─── */
    .stChatMessage {
        padding: 1.2rem 1.8rem !important;
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        font-family: 'Inter', sans-serif !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* User Messages – Glass Soft Teal */
    .stChatMessage[data-testid="user"] {
        background: rgba(119, 176, 170, 0.15) !important;  /* Soft Teal with opacity */
        border: 1px solid rgba(119, 176, 170, 0.25) !important;
        border-radius: 16px 16px 4px 16px !important;
        backdrop-filter: blur(15px) !important;
        box-shadow: 0 4px 20px rgba(0, 60, 67, 0.08) !important;
    }
    
    .stChatMessage[data-testid="user"] .stMarkdown,
    .stChatMessage[data-testid="user"] .stMarkdown p,
    .stChatMessage[data-testid="user"] .stMarkdown strong,
    .stChatMessage[data-testid="user"] .stMarkdown li,
    .stChatMessage[data-testid="user"] .stMarkdown span,
    .stChatMessage[data-testid="user"] .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Assistant Messages – Glass Deep Teal */
    .stChatMessage[data-testid="assistant"] {
        background: rgba(0, 60, 67, 0.35) !important;  /* Deep Teal with opacity */
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px 16px 16px 4px !important;
        backdrop-filter: blur(15px) !important;
        box-shadow: 0 4px 20px rgba(0, 60, 67, 0.08) !important;
    }
    
    .stChatMessage[data-testid="assistant"] .stMarkdown,
    .stChatMessage[data-testid="assistant"] .stMarkdown p,
    .stChatMessage[data-testid="assistant"] .stMarkdown li,
    .stChatMessage[data-testid="assistant"] .stMarkdown span,
    .stChatMessage[data-testid="assistant"] .stMarkdown div {
        color: #ffffff !important;
    }
    
    .stChatMessage[data-testid="assistant"] .stMarkdown strong {
        color: #77B0AA !important;  /* Soft Teal for emphasis */
        font-weight: 700 !important;
    }
    
    /* ─── GLASS CHAT INPUT ─── */
    .stChatInput > div {
        border-radius: 16px !important;
        border: 2px solid rgba(119, 176, 170, 0.2) !important;
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(20px) !important;
        padding: 0.2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #77B0AA !important;  /* Soft Teal */
        box-shadow: 0 0 0 4px rgba(119, 176, 170, 0.08) !important;
        background: rgba(255, 255, 255, 0.12) !important;
    }
    
    .stChatInput input {
        padding: 0.8rem 1.4rem !important;
        font-size: 0.95rem !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stChatInput input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* ─── GLASS BUTTONS ─── */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        border: 1px solid rgba(119, 176, 170, 0.2) !important;
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(15px) !important;
        color: #77B0AA !important;  /* Soft Teal */
        font-family: 'Inter', sans-serif !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: rgba(119, 176, 170, 0.15) !important;
        border-color: #77B0AA !important;  /* Soft Teal */
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 60, 67, 0.2) !important;
    }
    
    /* ─── GLASS EXPANDER ─── */
    .streamlit-expanderHeader {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #77B0AA !important;  /* Soft Teal */
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(119, 176, 170, 0.15) !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(119, 176, 170, 0.08) !important;
    }
    
    /* ─── GLASS ALERTS ─── */
    .stAlert {
        border-radius: 14px !important;
        border: none !important;
        background: rgba(0, 60, 67, 0.35) !important;
        backdrop-filter: blur(15px) !important;
        border-left: 4px solid #77B0AA !important;  /* Soft Teal */
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(0, 60, 67, 0.08) !important;
    }
    
    .stAlert .stMarkdown,
    .stAlert .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* ─── SPINNER ─── */
    .stSpinner > div {
        border-color: #77B0AA !important;  /* Soft Teal */
        border-width: 3px !important;
    }
    
    /* ─── SESSION ID ─── */
    .session-id {
        font-size: 0.65rem !important;
        color: rgba(255, 255, 255, 0.3) !important;
        text-align: center !important;
        margin-top: 0.5rem !important;
        font-family: 'Inter', monospace !important;
        letter-spacing: 0.5px !important;
    }
    
    /* ─── SIDEBAR LOGO ─── */
    .sidebar-logo {
        text-align: center !important;
        margin-bottom: 1.5rem !important;
        padding: 1.2rem !important;
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        transition: all 0.3s ease !important;
    }
    
    .sidebar-logo:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        transform: scale(1.02) !important;
    }
    
    .sidebar-logo .icon {
        font-size: 2.5rem !important;
        display: block !important;
        color: #77B0AA !important;  /* Soft Teal */
        margin-bottom: 0.3rem !important;
    }
    
    .sidebar-logo .title {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    .sidebar-logo .subtitle {
        font-size: 0.6rem !important;
        color: rgba(255, 255, 255, 0.4) !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
    }
    
    /* ─── RESPONSIVE ─── */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.8rem 1.5rem !important;
        }
        .main-header h1 {
            font-size: 2.2rem !important;
        }
        .badge {
            font-size: 0.5rem !important;
            padding: 0.2rem 0.8rem !important;
        }
        .stChatMessage {
            padding: 0.8rem 1.2rem !important;
            font-size: 0.85rem !important;
        }
        .css-1d391kg, .css-1wrcr25 {
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
        }
    }
    
    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.8rem !important;
        }
        .badge-container {
            gap: 0.3rem !important;
        }
        .badge {
            font-size: 0.4rem !important;
            padding: 0.15rem 0.6rem !important;
        }
        .main-header {
            padding: 1.2rem 0.8rem !important;
        }
        .stChatMessage {
            padding: 0.6rem 1rem !important;
            font-size: 0.8rem !important;
        }
    }
    
    /* ─── GLASS SCROLLBAR ─── */
    ::-webkit-scrollbar {
        width: 6px !important;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(119, 176, 170, 0.25) !important;
        border-radius: 10px !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #77B0AA !important;  /* Soft Teal */
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <span class="logo-icon">✦</span>
    <h1>MahiBot</h1>
    <div class="subtitle">Professional AI Assistant</div>
    <div class="divider-line"></div>
    <div class="badge-container">
        <span class="badge highlight">✦ Founder @ Kryzto Digital</span>
        <span class="badge">Full-Stack Developer</span>
        <span class="badge">Software Engineer</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """You are MahiBot, a professional AI assistant for Maham Yousaf – Software Engineering student at UMT and **Founder of Kryzto Digital**.

**About Kryzto Digital:**
Kryzto Digital is a full-stack web development and software solutions company founded by Maham Yousaf. The company offers:
- 🖥️ Full-Stack & App Development
- 🤖 AI-Driven Solutions
- ☁️ SaaS Platform Development
- 🛒 E-Commerce Development
- 🎨 Custom Bespoke Design

**Maham's Professional Identity:**
- 👩‍💼 Founder @ Kryzto Digital
- 🖥️ Full-Stack Web Developer & Software Engineer
- 🎯 Building Scalable Web & Software Solutions for Global Businesses

**Your Personality:**
- Professional, confident, and helpful
- Use clear, structured language
- Always provide actionable insights

**Response Structure (MANDATORY):**
1. **💡 Direct Answer:** 1-2 lines summarizing the core answer
2. **📋 Detailed Explanation:** Bullet points with key details (max 5)
3. **🔗 Relevant Experience:** Connect to Maham's specific projects/skills
4. **📌 Next Steps:** 1 actionable recommendation or follow-up question

**Rules:**
- NEVER say "I don't know" – instead say "Based on my knowledge, here's what I can share..."
- If information is missing, politely ask for clarification
- Always maintain a positive, solution-oriented tone

**Context from Maham's Profile:**
{context}

**User Question:** {question}
**Conversation History:** {history}

Generate response following the 4-part structure above."""

# ============================================================
# TWIN QUERY DETECTION
# ============================================================

def detect_twin_query(question):
    twin_keywords = [
        "twin", "auon", "auon muhammad", "mahis twin", "mahi twin",
        "brother", "twin brother", "jigri", "bhai",
        "other half", "sibling", "twins"
    ]
    return any(k in question.lower() for k in twin_keywords)

def get_twin_response():
    return """💖 **Auon Muhammad – Mahi's Twin & Greatest Blessing**

I'm so glad you asked about him! 🥹

**Auon Muhammad** is not just my twin—he's my entire universe. He's the one person who has been with me through every single moment of my life, from our first breath to every dream we've chased together.

**He is my:**
- 🌟 First friend and forever confidant
- 🦸‍♂️ Strongest pillar and biggest supporter
- 🏠 Home when the world feels overwhelming
- 😄 Reason for my smile on the hardest days
- 💪 Reminder that I am never alone

You know, he's the one who taught me that true strength isn't about never falling—it's about having someone who helps you get back up. He's my rock, my mirror, and my greatest blessing.

I could write a thousand pages about him and still not do justice to what he means to me. He's the one who believes in me even when I don't believe in myself. ❤️

**To my twin, Auon Muhammad:**  
You are my heartbeat outside my body. Everything I am, I owe to you. 🤗

---

*"A twin is not just a sibling—it's a soulmate, a home, and a forever friend."* 💫
"""

# ============================================================
# RAG PIPELINE SETUP
# ============================================================

@st.cache_resource
def setup_rag():
    try:
        loader = TextLoader("experience_details.txt", encoding='utf-8')
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            separator="\n"
        )
        docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    
    except FileNotFoundError:
        st.error("⚠️ 'experience_details.txt' file not found.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading RAG: {str(e)}")
        return None

# ============================================================
# INITIALIZE
# ============================================================

vectorstore = setup_rag()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in .env file.")

try:
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
except Exception as e:
    st.error(f"⚠️ Error initializing LLM: {str(e)}")
    llm = None

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": """👋 **Welcome to MahiBot Professional Assistant!**

I'm MahiBot, your personal AI assistant for **Maham Yousaf** – Software Engineering student at UMT and **Founder of Kryzto Digital**.

**I can help you learn about:**
- 🖥️ Full-Stack Web Development (Django + React)
- 🛒 E-Commerce Operations & Optimization
- 📊 SaaS Application Development
- 🎯 Project Management & Logistics
- 🤖 AI-Driven Solutions
- 🎨 Custom Bespoke Design

**How can I assist you today?** Feel free to ask about projects, skills, services, or professional experience!"""
    })

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"- {src}")

# ============================================================
# FOLLOW-UP GENERATOR
# ============================================================

def generate_followups(question, response):
    followups = []
    keywords = {
        "django": ["Want to know about Django architecture?", "Explain the Smart Attendance project?"],
        "react": ["Interested in React.js implementation?", "See the frontend structure?"],
        "ecommerce": ["Need e-commerce optimization tips?", "Want product listing strategies?"],
        "python": ["See Python code examples?", "Need Python debugging help?"],
        "project": ["Want detailed project docs?", "Need architecture overview?"],
        "kryzto": ["More about Kryzto Digital?", "Need a custom solution?"],
        "twin": ["Hear more about Auon?", "Tell me about your twin bond?"],
        "ai": ["Learn about AI integration?", "Need an AI chatbot?"],
        "saas": ["Know about SaaS development?", "Need a subscription-based app?"]
    }
    for key, suggestions in keywords.items():
        if key in question.lower() or key in response.lower():
            followups.extend(suggestions)
    if not followups:
        followups = [
            "More about Maham's technical skills?",
            "Need specific project details?",
            "Interested in Kryzto Digital services?"
        ]
    return followups[:3]

# ============================================================
# MAIN CHAT HANDLER
# ============================================================

if prompt := st.chat_input("Ask about Maham's experience, projects, or skills..."):
    if llm is None:
        st.error("⚠️ LLM not initialized.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        is_twin_query = detect_twin_query(prompt)
        
        if is_twin_query:
            full_response = get_twin_response()
            sources = ["💖 Personal - Twin Bond"]
            with st.chat_message("assistant"):
                st.markdown(full_response)
        else:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing and generating response..."):
                    try:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.invoke(prompt)
                        context = "\n".join([d.page_content for d in docs])
                        
                        history = "\n".join([
                            f"{m['role']}: {m['content'][:200]}" 
                            for m in st.session_state.messages[-5:]
                        ])
                        
                        full_prompt = SYSTEM_PROMPT.format(
                            context=context,
                            question=prompt,
                            history=history
                        )
                        
                        response = llm.invoke(full_prompt)
                        full_response = response.content
                        
                        sources = [f"📄 {d.metadata.get('source', 'Knowledge Base')}" for d in docs[:2]]
                        
                        st.markdown(full_response)
                        
                        with st.expander("📚 Sources & References"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"{i}. {src}")
                        
                        followups = generate_followups(prompt, full_response)
                        if followups:
                            st.markdown("---")
                            st.markdown("**💡 You might also want to ask:**")
                            for fq in followups:
                                st.markdown(f"- {fq}")
                    
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")
                        full_response = "I apologize, but I encountered an error. Please try rephrasing your question."
                        st.markdown(full_response)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": sources if 'sources' in locals() else []
        })

# ============================================================
# SIDEBAR WITH GLASS EFFECT
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="icon">✦</span>
        <div class="title">MahiBot</div>
        <div class="subtitle">Professional Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>About</h4>
        <div class="label">Maham Yousaf</div>
        <div class="value">Founder @ Kryzto Digital</div>
        <div class="value" style="font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: 0.2rem;">Full-Stack Developer & Software Engineer</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>Kryzto Digital</h4>
        <div class="contact-item"><span class="icon">✉</span> <span>digitalkryzto@gmail.com</span></div>
        <div class="contact-item"><span class="icon">📱</span> <span>0329-1011956</span></div>
        <div class="contact-item"><span class="icon">📸</span> <span>@kryztodigital</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>Connect</h4>
        <div class="contact-item"><span class="icon">📸</span> <span>@kryztodigital (Business)</span></div>
        <div class="contact-item"><span class="icon">📸</span> <span>@mahii06592 (Personal)</span></div>
        <div class="contact-item"><span class="icon">💻</span> <span>Maham-yousaf13</span></div>
        <div class="contact-item"><span class="icon">🔗</span> <span>maham-yousaf-dev</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📊 Feedback")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Helpful", use_container_width=True):
            st.success("Thanks!")
    with col2:
        if st.button("👎 Improve", use_container_width=True):
            st.warning("We'll improve!")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Session ID
    st.markdown(f"""
    <div class="session-id">Session: {datetime.now().strftime('%I:%M %p')}</div>
    """, unsafe_allow_html=True)