import streamlit as st
import re
import logging
import concurrent.futures
import time
import os
from typing import List
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from cachetools import LRUCache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions
def cache_with_limit(cache_dict, key, value):
    cache_dict[key] = value

def get_recent_conversation_context(messages, max_pairs=2):
    history = []
    for msg in messages[-(max_pairs * 2):]:
        role = "User" if msg["role"] == "user" else "Assistant"
        weight = 1.5 if msg["content"].lower().startswith(("why", "how", "what")) else 1.0
        history.append((f"{role}: {msg['content']}", weight))
    history.sort(key=lambda x: x[1], reverse=True)
    return "\n".join(h[0] for h in history) if history else ""

# Snowflake connection
@st.cache_resource
def create_snowflake_session():
    try:
        connection_parameters = {
            "account": st.secrets["account"],
            "user": st.secrets["user"],
            "password": st.secrets["password"],
            "warehouse": st.secrets["warehouse"],
            "database": st.secrets["database"],
            "schema": st.secrets["schema"],
            "role": st.secrets["role"],
        }
    except (KeyError, TypeError):
        connection_parameters = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT", "fokiamm-yqb60913"),
            "user": os.getenv("SNOWFLAKE_USER", "streamlit_demo_user"),
            "password": os.getenv("SNOWFLAKE_PASSWORD", "RagCortex#78_Pw"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "CORTEX_SEARCH_TUTORIAL_WH"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "CORTEX_SEARCH_TUTORIAL_DB"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
            "role": os.getenv("SNOWFLAKE_ROLE", "STREAMLIT_READONLY_ROLE"),
        }
        st.error("Using fallback environment variables for Snowflake.")
    return Session.builder.configs(connection_parameters).create()

session = create_snowflake_session()
root = Root(session)
search_service = root.databases["CORTEX_SEARCH_TUTORIAL_DB"].schemas["PUBLIC"].cortex_search_services["FOMC_SEARCH_SERVICE"]

# Utility functions
def extract_target_years(query: str) -> List[int]:
    return [int(y) for y in re.findall(r"20\d{2}", query)]

def extract_file_year(file_name: str) -> int:
    match = re.search(r"(\d{4})", file_name)
    return int(match.group(1)) if match else 0

def clean_chunk(chunk: str) -> str:
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", chunk)
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def extract_clean_title(file_name: str) -> str:
    month_map = {
        "01": "January", "02": "February", "03": "March", "04": "April",
        "05": "May", "06": "June", "07": "July", "08": "August",
        "09": "September", "10": "October", "11": "November", "12": "December",
    }
    match = re.search(r"(\d{4})(\d{2})(\d{2})", file_name)
    if match:
        year, month, day = match.groups()
        date_str = f"{month_map.get(month, month)} {int(day)}, {year}"
    else:
        date_str = "Unknown Date"
    if "minutes" in file_name.lower():
        doc_type = "FOMC Minutes"
    elif "beigebook" in file_name.lower():
        doc_type = "Beige Book"
    else:
        doc_type = "FOMC Document"
    return f"{doc_type} - {date_str}"

def create_direct_link(file_name: str) -> str:
    base = "https://www.federalreserve.gov/monetarypolicy/files/"
    name = file_name.split("/")[-1]
    return f"{base}{name}"

class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit: int = 12):
        self._service = search_service
        self.limit = limit
    def retrieve(self, query: str) -> List[dict]:
        raw_results = self._service.search(query=query, columns=["chunk", "file_name"], limit=150).results
        seen = {}
        for r in raw_results:
            if r["file_name"] not in seen:
                seen[r["file_name"]] = r
        return list(seen.values())[:self.limit]

rag_retriever = CortexSearchRetriever(session)

# PDF generator
def create_pdf(history_md: str) -> BytesIO:
    history_md = re.sub(r'\[Link\]\((.*?)\)', r'\1', history_md)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for line in history_md.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))
    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit setup
st.set_page_config(page_title="Chat with the Federal Reserve", page_icon="💬", layout="centered")
st.title("💬 Chat with the Federal Reserve - Enhanced Conversational Mode")
st.markdown("**Supports multi-document reasoning and FOMC research context.**")
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = LRUCache(maxsize=20)
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []

# Render chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖").markdown(msg["content"], unsafe_allow_html=False)

# Chat input
user_input = st.chat_input("Ask the Fed about policy, inflation, outlooks, or Beige Book insights...")
if user_input:
    st.chat_message("user", avatar="👤").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": "Simulated response for debugging UI layout."})

# Buttons directly below input
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Conversation"):
        st.session_state.messages.clear()
        st.session_state.rag_cache.clear()
        st.session_state.last_contexts.clear()
        st.rerun()
with col2:
    current_time = datetime.now().strftime("%B %d, %Y %H:%M:%S")
    history_md = f"# Chat History - {current_time}\n\n"
    for msg in st.session_state.messages:
        history_md += f"**{msg['role'].capitalize()}**: {msg['content']}\n\n"
    if st.session_state.last_contexts:
        history_md += "## Sources Used in Last Response\n\n"
        for c in st.session_state.last_contexts:
            title = extract_clean_title(c["file_name"])
            pdf_url = create_direct_link(c["file_name"])
            snippet = clean_chunk(c["chunk"])[:350] + ("..." if len(c["chunk"]) > 350 else "")
            history_md += f"- **{title}** ({pdf_url})\n {snippet}\n\n"
    else:
        history_md += "## Sources\n\nNo documents found for the last query.\n"
    pdf_buffer = create_pdf(history_md)
    st.download_button("📥 Download Chat History", pdf_buffer, "chat_history.pdf", "application/pdf")

# Suggested follow-ups always visible
if st.session_state.messages:
    follow_ups = ["Why were rates adjusted?", "What are the projections for next year?"]
    st.write("Suggested follow-ups:")
    for suggestion in follow_ups:
        if st.button(suggestion):
            st.chat_message("user", avatar="👤").write(suggestion)
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.session_state.messages.append({"role": "assistant", "content": "Simulated follow-up answer."})
