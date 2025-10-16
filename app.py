import streamlit as st
import re
import logging
import concurrent.futures
import time
from typing import List
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# ======================================================
# 🔧 INITIAL SETUP
# ======================================================

st.set_page_config(
    page_title="Chat with the Federal Reserve",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Chat with the Federal Reserve - Enhanced Conversational Mode")
st.markdown("**Supports multi-document reasoning, trend analysis, and Fed jargon explanation.**")

# Hide Streamlit default menu and footer for cleaner UI
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state keys
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = {}

# ======================================================
# 🔧 CACHE & MESSAGE MANAGEMENT HELPERS
# ======================================================

def cache_with_limit(cache_dict, key, value, max_size=20):
    """Add to cache with size limit (FIFO eviction)"""
    if len(cache_dict) >= max_size and key not in cache_dict:
        # Remove oldest entry
        first_key = next(iter(cache_dict))
        cache_dict.pop(first_key)
    cache_dict[key] = value

def get_recent_conversation_context(messages, max_pairs=3):
    """
    Get only the last N user/assistant pairs for conversation context.
    This prevents the prompt from growing too large.
    """
    user_assistant = [m for m in messages if m["role"] in ["user", "assistant"]]
    recent = user_assistant[-(max_pairs * 2):]
    
    # Format as conversation history string
    history = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {msg['content']}")
    
    return "\n".join(history) if history else ""

# ======================================================
# ❄️ SNOWFLAKE CONNECTION
# ======================================================

@st.cache_resource
def create_snowflake_session():
    connection_parameters = {
        "account": "fokiamm-yqb60913",
        "user": "streamlit_demo_user",
        "password": "RagCortex#78_Pw",
        "warehouse": "CORTEX_SEARCH_TUTORIAL_WH",
        "database": "CORTEX_SEARCH_TUTORIAL_DB",
        "schema": "PUBLIC",
        "role": "STREAMLIT_READONLY_ROLE",
    }
    return Session.builder.configs(connection_parameters).create()


session = create_snowflake_session()
root = Root(session)
search_service = (
    root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
    .schemas["PUBLIC"]
    .cortex_search_services["FOMC_SEARCH_SERVICE"]
)

# ======================================================
# 🧹 TEXT & FILE HELPERS
# ======================================================

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
    match = re.search(r"(\d{4})(\d{2})(\d{2})", file_name)
    month_map = {
        "01": "January", "02": "February", "03": "March", "04": "April",
        "05": "May", "06": "June", "07": "July", "08": "August",
        "09": "September", "10": "October", "11": "November", "12": "December",
    }

    if match:
        year, month, day = match.groups()
        date_str = f"{month_map.get(month, month)} {int(day)}, {year}"
    else:
        date_str = "Unknown Date"

    fname = file_name.lower()
    if "beigebook" in fname or "beigebook_" in fname:
        doc_type = "Beige Book"
    elif "longerungoals" in fname or "fomc_longerungoals" in fname:
        doc_type = "FOMC Longer-Run Goals"
    elif "presconf" in fname or "fomcpresconf" in fname:
        doc_type = "Press Conference"
    elif "projtabl" in fname or "fomcprojtabl" in fname:
        doc_type = "Projection Tables"
    elif "mprfullreport" in fname or "mpr" in fname:
        doc_type = "Monetary Policy Report"
    elif "monetary" in fname:
        doc_type = "Monetary Document"
    elif "financial-stability-report" in fname or "financial" in fname:
        doc_type = "Financial Stability Report"
    elif "minutes" in fname or "fomcminutes" in fname:
        doc_type = "FOMC Minutes"
    else:
        doc_type = "FOMC Document"

    return f"{doc_type} - {date_str}"


def create_direct_link(file_name: str) -> str:
    """
    Build the correct public URL for any Federal Reserve PDF
    based on its filename pattern.
    """
    try:
        base = "https://www.federalreserve.gov"
        name = file_name.split("/")[-1]

        mapping = [
            (r"beigebook", f"{base}/monetarypolicy/files/"),
            (r"fomc_longerungoals", f"{base}/monetarypolicy/files/"),
            (r"fomc_longerungoals_2023", f"{base}/monetarypolicy/files/"),
            (r"fomc_longerungoals_2024", f"{base}/monetarypolicy/files/"),
            (r"fomcprojtabl", f"{base}/monetarypolicy/files/"),
            (r"fomcpresconf", f"{base}/mediacenter/files/"),
            (r"presconf", f"{base}/mediacenter/files/"),
            (r"monetary", f"{base}/monetarypolicy/files/"),
            (r"financial-stability-report", f"{base}/publications/files/"),
            (r"mprfullreport", f"{base}/monetarypolicy/files/"),
            (r"fomcminutes", f"{base}/monetarypolicy/files/"),
        ]

        lower = name.lower()
        for pattern, prefix in mapping:
            if pattern in lower:
                return prefix + name

        # default fallback to monetarypolicy files
        return f"{base}/monetarypolicy/files/{name}"
    except Exception as e:
        logging.error(f"create_direct_link failed for {file_name}: {e}")
        return f"https://www.federalreserve.gov/monetarypolicy/files/{file_name.split('/')[-1]}"

# ======================================================
# 🔍 RETRIEVER CLASS
# ======================================================

class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit: int = 20):
        self._root = Root(snowpark_session)
        self._service = search_service
        self.limit = limit

    def retrieve(self, query: str) -> List[dict]:
        try:
            raw_results = self._service.search(
                query=query,
                columns=["chunk", "file_name"],
                limit=150
            ).results
            unique_docs = {}
            for r in raw_results:
                if r["file_name"] not in unique_docs:
                    unique_docs[r["file_name"]] = r
            docs = list(unique_docs.values())

            target_years = extract_target_years(query)
            if target_years:
                lower_year = min(target_years) - 1
                upper_year = max(target_years)
                filtered_docs = [d for d in docs if lower_year <= extract_file_year(d["file_name"]) <= upper_year]
                if filtered_docs:
                    docs = filtered_docs

            docs = sorted(docs, key=lambda d: extract_file_year(d["file_name"]), reverse=True)
            docs = docs[:self.limit]

            return [{"chunk": d["chunk"], "file_name": d["file_name"]} for d in docs]

        except Exception as e:
            logging.error(f"Cortex Search retrieval error: {e}")
            st.error(f"❌ Cortex Search Error: {e}")
            return []

rag_retriever = CortexSearchRetriever(session)

# ======================================================
# 📘 PROMPT GENERATION (OPTIMIZED)
# ======================================================

glossary = """
Glossary:
- Dot Plot: A chart showing each FOMC participant's forecast for the federal funds rate.
- Longer-run Inflation Expectations: Fed members' inflation expectations beyond the near-term future.
- Beige Book: A report summarizing economic conditions across Fed districts, published 8 times/year.
- Federal Funds Rate Target: The interest rate that the Fed targets for overnight lending between banks.
"""

def build_system_prompt(query: str, contexts: List[dict], conversation_history: str = "") -> str:
    """
    Build an optimized system prompt that doesn't grow with conversation length.
    Only includes the current query's context + brief conversation history.
    """
    # Group contexts by year (limit to top 5 to reduce size)
    year_buckets = {}
    for c in contexts[:5]:  # Reduced from 7 to 5
        year = extract_file_year(c["file_name"])
        if year not in year_buckets:
            year_buckets[year] = []
        year_buckets[year].append(clean_chunk(c["chunk"]))
    
    grouped_texts = []
    for year in sorted(year_buckets.keys()):
        grouped_texts.append(f"Year {year} excerpts:\n{chr(10).join(year_buckets[year])}")
    
    context_text = "\n\n".join(grouped_texts)
    
    # Strict token limit on context
    if len(context_text) > 2500:  # Reduced from 3500
        context_text = context_text[:2500]

    # Add conversation history if available (keeps context between questions)
    history_section = ""
    if conversation_history:
        history_section = f"\n\nRecent conversation:\n{conversation_history}\n"

    prompt = f"""You are an expert economic analyst specializing in Federal Reserve communications.

Today is {datetime.now():%B %d, %Y}.

{glossary}

Use ONLY the following excerpts from FOMC documents to answer the user's question. Do not invent facts. When relevant, cite the document type and year (e.g., "According to the January 2025 FOMC Minutes...").

Context excerpts by year:

{context_text}
{history_section}
User Question: {query}

Answer:"""
    
    return prompt

# ======================================================
# 🧠 LLM COMPLETION (OPTIMIZED)
# ======================================================

def retrieve_with_timeout(query: str, timeout: float = 25.0, retries: int = 1) -> List[dict]:
    """
    Wrap rag_retriever.retrieve with timeout + retry + caching.
    """
    if not query:
        return []

    # Check cache first
    if query in st.session_state.rag_cache:
        return st.session_state.rag_cache[query]

    def _call():
        return rag_retriever.retrieve(query)

    for attempt in range(retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_call)
                results = future.result(timeout=timeout)
                # Cache with size limit
                cache_with_limit(st.session_state.rag_cache, query, results, max_size=20)
                return results
        except concurrent.futures.TimeoutError:
            logging.warning(f"Retrieval timed out (attempt {attempt+1}/{retries+1})")
            time.sleep(1)
        except Exception as e:
            logging.error(f"Retrieval error (attempt {attempt+1}/{retries+1}): {e}")
            time.sleep(1)

    # Fallback to cached result if available
    fallback = st.session_state.rag_cache.get(query, [])
    if fallback:
        logging.warning("Using cached retrieval results as fallback.")
    return fallback

def generate_response_stream(query: str, contexts: List[dict], conversation_history: str = ""):
    """
    Streaming call with retries and fallback.
    """
    prompt = build_system_prompt(query, contexts, conversation_history)

    def run_completion():
        return complete("claude-3-5-sonnet", prompt, stream=True, session=session)

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_completion)
                return future.result(timeout=60)  # Reduced from 90
        except concurrent.futures.TimeoutError:
            logging.warning(f"Cortex response timed out (attempt {attempt+1}/{max_retries+1})")
            time.sleep(2)
        except Exception as e:
            logging.error(f"Cortex streaming error (attempt {attempt+1}/{max_retries+1}): {e}")
            time.sleep(2)

    # Silent fallback to non-streaming
    try:
        logging.warning("Falling back to non-streaming completion.")
        backup = complete("claude-3-5-sonnet", prompt, session=session)
        return iter([backup])
    except Exception as e:
        logging.error(f"Backup completion failed: {e}")
        return iter(["I apologize, but I'm having trouble generating a response right now. Please try again."])

# ======================================================
# 💬 STREAMLIT UI LOGIC
# ======================================================

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()
    st.session_state.rag_cache.clear()
    st.rerun()

# Display chat history (only user/assistant messages)
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        st.chat_message(msg["role"]).write(msg["content"])

def run_query(user_query: str):
    """
    Main query execution - retrieves context and generates response.
    KEY FIX: Does NOT store system prompts in session state.
    """
    # Get recent conversation for context
    conversation_history = get_recent_conversation_context(st.session_state.messages, max_pairs=2)
    
    # Retrieve context with timeout
    with st.spinner("Searching..."):
        contexts = retrieve_with_timeout(user_query, timeout=25.0, retries=1)

    if not contexts:
        st.info("No relevant context found. Answering from general knowledge.")

    # Generate response (system prompt is NOT saved to session state)
    with st.spinner("Generating response..."):
        stream = generate_response_stream(user_query, contexts, conversation_history)

    # Stream and render assistant response
    response_text = ""
    assistant_container = st.chat_message("assistant", avatar="🤖")
    placeholder = assistant_container.empty()
    
    for token in stream:
        try:
            response_text += token
            placeholder.markdown(response_text)
        except Exception:
            logging.exception("Error while streaming chunk")

    # Store ONLY user and assistant messages (NOT system prompts)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Keep message history manageable (limit to last 10 messages)
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]

    # Show context sources
    top_contexts = contexts[:3] if contexts else []
    if top_contexts:
        with st.expander("📄 View Context (top 3)"):
            for c in top_contexts:
                title = extract_clean_title(c["file_name"])
                pdf_url = create_direct_link(c["file_name"])
                snippet = clean_chunk(c["chunk"])[:350]
                if len(c["chunk"]) > 350:
                    snippet += "..."
                st.markdown(f"**[{title}]({pdf_url})**")
                st.caption(snippet)
                st.divider()

# Chat input
user_input = st.chat_input("Ask the Fed about policy, inflation, outlooks, or Beige Book insights...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    run_query(user_input)
