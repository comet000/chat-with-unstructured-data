import streamlit as st
import logging
import concurrent.futures
import time
from typing import List
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Chat with the Federal Reserve", page_icon="💬", layout="centered")
st.title("🏦 Chat with the Federal Reserve")
st.markdown("**Built on 5,000 pages of Federal Reserve documents (2023–2025)**")

# Hide Streamlit default menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Initialize state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = {}

# --- Snowflake Session ---
@st.cache_resource
def create_snowflake_session():
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
    }
    return Session.builder.configs(connection_parameters).create()

session = create_snowflake_session()

# --- PDF URL Helper ---
def create_direct_link(file_name: str) -> str:
    """
    Build the correct public URL for any Federal Reserve PDF
    based on its filename pattern.
    """
    base = "https://www.federalreserve.gov"
    mapping = [
        (r"BeigeBook_", f"{base}/monetarypolicy/files/BeigeBook_"),
        (r"FOMC_LongerRunGoals", f"{base}/monetarypolicy/files/FOMC_LongerRunGoals"),
        (r"fomcprojtabl", f"{base}/monetarypolicy/files/"),
        (r"FOMCpresconf", f"{base}/mediacenter/files/"),
        (r"fomcpresconf", f"{base}/mediacenter/files/"),
        (r"fomcminutes", f"{base}/monetarypolicy/files/"),
        (r"monetary", f"{base}/monetarypolicy/files/"),
        (r"financial-stability-report", f"{base}/publications/files/"),
        (r"mprfullreport", f"{base}/monetarypolicy/files/"),
    ]
    for pattern, prefix in mapping:
        if pattern.lower() in file_name.lower():
            return prefix + file_name.split("/")[-1]
    return f"{base}/monetarypolicy/files/{file_name.split('/')[-1]}"

# --- RAG Retrieval ---
class RAGRetriever:
    def __init__(self, session):
        self.session = session
        self.root = Root(session)
        self.collection = self.root.databases[session.get_current_database()] \
            .schemas[session.get_current_schema()] \
            .vector_collections["fed_rag_collection"]

    def retrieve(self, query: str) -> List[dict]:
        try:
            docs = self.collection.search(query, columns=["chunk", "file_name"], limit=5)
            return [{"chunk": d["chunk"], "file_name": d["file_name"]} for d in docs]
        except Exception as e:
            logging.error(f"Cortex Search retrieval error: {e}")
            return []

rag_retriever = RAGRetriever(session)

# --- System Prompt Builder ---
def build_system_prompt(query: str, contexts: List[dict]) -> str:
    combined_context = "\n\n".join([c["chunk"] for c in contexts])
    return f"""
You are an expert Federal Reserve analyst. Use the context below to answer the user’s question.

Context:
{combined_context}

Question:
{query}

Provide a clear, factual, and concise answer referencing relevant documents.
    """

# --- Safe Cortex Completion with Timeout and Fallback ---
def generate_response_stream(query: str, contexts: List[dict]):
    prompt = build_system_prompt(query, contexts)

    def run_completion():
        return complete("claude-3-5-sonnet", prompt, stream=True, session=session)

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_completion)
                return future.result(timeout=90)
        except concurrent.futures.TimeoutError:
            logging.warning(f"Cortex response timed out (attempt {attempt+1}/{max_retries}). Retrying...")
            time.sleep(2)
        except Exception as e:
            logging.error(f"Cortex streaming error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)

    # Silent fallback
    try:
        logging.warning("Falling back to non-streaming completion mode.")
        backup = complete("claude-3-5-sonnet", prompt, session=session)
        return iter([backup])
    except Exception as e:
        logging.error(f"Backup completion failed: {e}")
        return iter([])

# --- Streamlit Chat Logic ---
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask something about recent Fed policy, inflation, or FOMC decisions...")

def run_query(user_query: str):
    with st.spinner("Searching relevant Fed documents..."):
        contexts = rag_retriever.retrieve(user_query)

    with st.expander("🔍 See Retrieved Context"):
        if not contexts:
            st.info("No relevant documents found.")
        for item in contexts:
            pdf_url = create_direct_link(item["file_name"])
            snippet = item["chunk"][:600]
            snippet += "..." if len(item["chunk"]) > 600 else ""
            st.markdown(f"**📄 [{item['file_name'].split('/')[-1]}]({pdf_url})**")
            st.write(snippet)

    if not contexts:
        return ["No relevant context found."]

    st.session_state.messages.append({"role": "system", "content": build_system_prompt(user_query, contexts)})

    with st.spinner("Generating response..."):
        stream = generate_response_stream(user_query, contexts)
        response_text = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in stream:
                response_text += chunk
                placeholder.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    run_query(user_query)
