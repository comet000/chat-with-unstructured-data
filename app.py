import streamlit as st
import logging
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Chat with the Federal Reserve",
    page_icon="💬",
    layout="centered"
)

st.title("🏦 Chat with the Federal Reserve")
st.markdown("**Built on 5000 pages of Fed documents from 2023 - 2025**")

# Hide Streamlit default menu and footer for cleaner UI
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Initialize session state variables safely ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = {}

# --- Snowflake session creation ---
@st.cache_resource
def create_snowflake_session():
    connection_parameters = {
        # TODO: add your connection details here
        # "account": "your_account",
        # "user": "your_user",
        # "password": "your_password",
        # "warehouse": "your_warehouse",
        # "database": "your_database",
        # "schema": "your_schema",
    }
    try:
        session = Session.builder.configs(connection_parameters).create()
        return session
    except Exception as e:
        st.error(f"❌ Error creating Snowflake session: {e}")
        logging.error(f"Snowflake connection error: {e}")
        return None


# --- RAG Retriever Class Example ---
class RAGRetriever:
    def __init__(self, session):
        self.session = session

    def retrieve(self, query: str):
        try:
            # Example: implement Cortex Search here
            docs = []  # Placeholder for search results
            return [{"chunk": d["chunk"], "file_name": d["file_name"]} for d in docs]
        except Exception as e:
            logging.error(f"Cortex Search retrieval error: {e}")
            st.error(f"❌ Cortex Search Error: {e}")
            return []


# --- Chat Response Generation ---
def generate_response_stream(query: str, contexts: list):
    st.session_state.rag_cache.clear()

    for msg in st.session_state.messages:
        if msg["role"] != "system":  # Hide system messages from user
            st.chat_message(msg["role"]).write(msg["content"])


def run_query(user_query: str):
    with st.spinner("Searching..."):
        contexts = rag_retriever.retrieve(user_query)

    with st.expander("🔍 See Retrieved Context"):
        if not contexts:
            st.info("No relevant context found.")
        else:
            for item in contexts:
                title = extract_clean_title(item["file_name"])
                pdf_url = create_direct_link(item["file_name"])
                snippet = clean_chunk(item["chunk"])[:600]
                snippet += "..." if len(item["chunk"]) > 600 else ""
                st.markdown(f"**{title}**")
                st.write(snippet)
                st.markdown(f"[📄 View Full Document]({pdf_url})")

    if not contexts:
        st.info("No relevant documents found.")
        return ["No relevant context found."]

    st.session_state.messages.append({
        "role": "system",
        "content": build_system_prompt(user_query, contexts)
    })

    with st.spinner("Generating response..."):
        # Call LLM/Cortex completion here
        pass


# --- Initialize ---
session = create_snowflake_session()
if session:
    rag_retriever = RAGRetriever(session)
