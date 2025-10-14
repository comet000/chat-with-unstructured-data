import streamlit as st
import re
from typing import List
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

st.set_page_config(page_title="Chat with the Federal Reserve", page_icon="💬", layout="centered")
st.title("💬 Chat with the Federal Reserve")
st.markdown("**RAG chat on all FOMC PDFs between 2023 - 2025**")

# ------------------------
# Snowflake Session Setup
# ------------------------
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
search_service = root.databases["CORTEX_SEARCH_TUTORIAL_DB"].schemas["PUBLIC"].cortex_search_services["FOMC_SEARCH_SERVICE"]

# ------------------------
# Utility Functions
# ------------------------
def extract_target_years(query: str) -> List[int]:
    years = re.findall(r'20\d{2}', query)
    return [int(y) for y in years]

def extract_file_year(file_name: str) -> int:
    m = re.search(r'(\d{4})', file_name)
    return int(m.group(1)) if m else 0

def clean_chunk(chunk: str) -> str:
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)  # Remove images if any
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)  # Remove markdown headers
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize whitespace
    return cleaned

def extract_clean_title(file_name: str) -> str:
    m = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
    month_map = {
        '01':'January','02':'February','03':'March','04':'April',
        '05':'May','06':'June','07':'July','08':'August',
        '09':'September','10':'October','11':'November','12':'December'
    }
    if m:
        year, month, day = m.groups()
        month_name = month_map.get(month, month)
        date_str = f"{month_name} {int(day)}, {year}"
    else:
        date_str = "Unknown Date"

    if 'minutes' in file_name.lower():
        doc_type = "FOMC Minutes"
    elif 'proj' in file_name.lower():
        doc_type = "Summary of Economic Projections"
    elif 'presconf' in file_name.lower():
        doc_type = "Press Conference"
    elif 'beigebook' in file_name.lower():
        doc_type = "Beige Book"
    else:
        doc_type = "FOMC Document"

    return f"{doc_type} - {date_str}"

def create_direct_link(file_name: str) -> str:
    base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    if 'presconf' in file_name.lower():
        base_url = "https://www.federalreserve.gov/mediacenter/files/"
    elif 'beigebook' in file_name.lower():
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    return f"{base_url}{file_name}"

# ------------------------
# Retriever with Temporal Filtering
# ------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit: int = 15):
        self._root = Root(snowpark_session)
        self._service = search_service
        self.limit = limit

    def retrieve(self, query: str) -> List[dict]:
        try:
            raw_results = self._service.search(query=query, columns=["chunk", "file_name"], limit=100).results
            # Deduplicate documents by file_name
            unique_docs = {}
            for r in raw_results:
                if r["file_name"] not in unique_docs:
                    unique_docs[r["file_name"]] = r
            docs = list(unique_docs.values())

            # Extract target years from query
            target_years = extract_target_years(query)
            if target_years:
                # Filter documents by year proximity to any target year +-1 range
                lower_year = min(target_years) - 1
                upper_year = max(target_years)
                filtered_docs = [d for d in docs if lower_year <= extract_file_year(d["file_name"]) <= upper_year]
                if filtered_docs:
                    docs = filtered_docs  # Use filtered if non-empty

            # Sort final results by date descending to prioritize recent first
            docs = sorted(docs, key=lambda d: extract_file_year(d["file_name"]), reverse=True)
            # Limit results to retriever limit
            docs = docs[:self.limit]

            return [{"chunk": d["chunk"], "file_name": d["file_name"]} for d in docs]

        except Exception as e:
            st.error(f"❌ Error in Cortex Search retrieval: {e}")
            return []

# ------------------------
# RAG Chat Logic
# ------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)
        self.cache = st.session_state.setdefault("rag_cache", {})

    def retrieve_context(self, query: str) -> List[dict]:
        if query in self.cache:
            return self.cache[query]
        chunks = self.retriever.retrieve(query)
        self.cache[query] = chunks
        return chunks

    def build_prompt(self, query: str, contexts: List[dict]) -> str:
        joined_chunks = "\n\n".join([clean_chunk(c["chunk"]) for c in contexts[:3]])
        if len(joined_chunks) > 3500:
            joined_chunks = joined_chunks[:3500]

        additional_instructions = ""
        if any(keyword in query.lower() for keyword in ['outlook', 'projection', 'forecast', '2025', '2026']):
            additional_instructions = (
                "Include specific numerical or qualitative details about GDP, inflation, unemployment, "
                "and monetary policy stance where available.\n"
            )
        prompt = (
            f"You are an expert economic analyst familiar with FOMC documents.\n"
            f"Current date: {datetime.now():%B %d, %Y}.\n"
            f"{additional_instructions}"
            f"Use only the information provided in the following excerpts to answer the question below. "
            f"Do not guess or hallucinate any facts.\n\n"
            f"Context:\n{joined_chunks}\n\nQuestion: {query}\n\nAnswer:"
        )
        return prompt

    def generate_stream(self, query: str, contexts: List[dict]):
        prompt = self.build_prompt(query, contexts)
        return complete("claude-3-5-sonnet", prompt, stream=True, session=session)

rag = RAG()

# ------------------------
# Streamlit App Execution
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()
    st.session_state.rag_cache.clear()

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

def run_query(user_query: str):
    with st.spinner("Searching..."):
        contexts = rag.retrieve_context(user_query)

    with st.expander("🔍 See Retrieved Context"):
        if not contexts:
            st.info("No relevant context found.")
        for item in contexts:
            title = extract_clean_title(item["file_name"])
            pdf_url = create_direct_link(item["file_name"])
            snippet = clean_chunk(item["chunk"])[:600] + ("..." if len(item["chunk"]) > 600 else "")
            st.markdown(f"**{title}**")
            st.write(snippet)
            st.markdown(f"[📄 View Full Document]({pdf_url})")

    if not contexts:
        return ["No relevant documents found."]

    msgs = st.session_state.messages[:]
    msgs.append({"role": "system", "content": rag.build_prompt(user_query, contexts)})
    with st.spinner("Generating response..."):
        return rag.generate_stream(user_query, contexts)


user_input = st.chat_input("Ask about FOMC policy, Beige Book takeaways, or economic outlooks...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    stream = run_query(user_input)
    final = st.chat_message("assistant", avatar="🤖").write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": final})
