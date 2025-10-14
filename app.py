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

# --------------------------------------------------
# Snowflake Session
# --------------------------------------------------
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

# --------------------------------------------------
# Retriever with Date + Domain Logic
# --------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit: int = 10):
        self._root = Root(snowpark_session)
        self._service = search_service
        self.limit = limit

    def retrieve(self, query: str) -> List[dict]:
        results = self._service.search(query=query, columns=["chunk","file_name"], limit=self.limit).results
        unique = {}
        for r in results:
            if r["file_name"] not in unique:
                unique[r["file_name"]] = r
        sorted_res = self._intelligent_sort(list(unique.values()), query)
        return [{"chunk": r["chunk"], "file_name": r["file_name"]} for r in sorted_res]

    def _intelligent_sort(self, results, query):
        ql = query.lower()
        if any(t in ql for t in ["most recent","latest","newest","last"]):
            return self._sort_by_date(results, reverse=True)
        if "beige book" in ql and results:
            return self._sort_by_date(results, reverse=True)
        year_match = re.search(r'20[2-3][0-9]', ql)
        if year_match:
            return self._sort_by_relevance_to_year(results, int(year_match.group()))
        if any(t in ql for t in ["outlook","projection","forecast","future"]):
            return self._sort_by_date(results, reverse=True)
        return self._sort_by_date(results, reverse=True)

    def _sort_by_date(self, results, reverse=True):
        def extract(file_name):
            m = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
            return int("".join(m.groups())) if m else 0
        return sorted(results, key=lambda r: extract(r["file_name"]), reverse=reverse)

    def _sort_by_relevance_to_year(self, results, year):
        def score(file_name):
            m = re.search(r'(\d{4})', file_name)
            fy = int(m.group(1)) if m else 0
            return 100 if fy == year else (50 if abs(fy-year)==1 else 0)
        return sorted(results, key=lambda r: score(r["file_name"]), reverse=True)

# --------------------------------------------------
# Utility
# --------------------------------------------------
def extract_clean_title(fname):
    m = re.search(r'(\d{4})(\d{2})(\d{2})', fname)
    month_map = {'01':'January','02':'February','03':'March','04':'April','05':'May','06':'June',
                 '07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}
    date_str = f"{month_map[m.group(2)]} {int(m.group(3))}, {m.group(1)}" if m else "Unknown Date"
    if 'minutes' in fname.lower(): doc_type = "FOMC Minutes"
    elif 'proj' in fname.lower(): doc_type = "Summary of Economic Projections"
    elif 'presconf' in fname.lower(): doc_type = "Press Conference"
    elif 'beigebook' in fname.lower(): doc_type = "Beige Book"
    else: doc_type = "FOMC Document"
    return f"{doc_type} - {date_str}"

def create_direct_link(fname):
    return f"https://www.federalreserve.gov/monetarypolicy/files/{fname}"

def clean_chunk(chunk):
    return re.sub(r'\s+',' ', re.sub(r'#{1,6}\s*','', chunk)).strip()

# --------------------------------------------------
# RAG Core
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)
        self.cache = st.session_state.setdefault("rag_cache", {})

    def retrieve_context(self, query):
        if query in self.cache: return self.cache[query]
        chunks = self.retriever.retrieve(query)

        # Special case: Beige Book – ensure National Summary if missing
        if "beige book" in query.lower() and not any("National Summary" in c["chunk"] for c in chunks):
            bb_res = self.retriever.retrieve("Beige Book National Summary")
            if bb_res: chunks = bb_res

        self.cache[query] = chunks
        return chunks

    def summarize_context(self, query, contexts):
        if not contexts: return "No relevant context retrieved."
        key_details_clause = ""
        if any(k in query.lower() for k in ["outlook","projection","2026"]):
            key_details_clause = "Extract GDP growth, unemployment rate, inflation rate, and policy stance if present."

        ctx_txt = "\n\n".join([c["chunk"] for c in contexts[:3]])
        if len(ctx_txt) > 4000: ctx_txt = ctx_txt[:4000] + "..."
        prompt = f"You are an expert economist. Summarize clearly:\n{key_details_clause}\nContext:\n{ctx_txt}\nSummary:"
        return str(complete("claude-3-5-sonnet", prompt, session=session)).strip()

    def build_messages_with_context(self, msgs, query, ctx):
        sys_content = f"Current date: {datetime.now():%B %d, %Y}\n\nContext Summary:\n{self.summarize_context(query, ctx)}"
        return msgs + [{"role":"system","content":sys_content}]

    def generate_stream(self, msgs):
        return complete("claude-3-5-sonnet", msgs, stream=True, session=session)

rag = RAG()

# --------------------------------------------------
# UI Execution
# --------------------------------------------------
if "messages" not in st.session_state: st.session_state.messages = []

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()
    st.session_state.rag_cache.clear()

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

def run_query(q):
    with st.spinner("Searching..."):
        ctx = rag.retrieve_context(q)
    with st.expander("🔍 See Retrieved Context"):
        if not ctx: st.info("No relevant context found.")
        for item in ctx:
            st.markdown(f'**{extract_clean_title(item["file_name"])}**')
            st.write(clean_chunk(item["chunk"])[:600] + ("..." if len(item["chunk"])>600 else ""))
            st.markdown(f"[📄 View Full Document]({create_direct_link(item['file_name'])})")
    if not ctx: return ["No relevant docs."]
    msgs = rag.build_messages_with_context(st.session_state.messages, q, ctx)
    with st.spinner("Generating response..."):
        return rag.generate_stream(msgs)

user_in = st.chat_input("Ask about FOMC policy, Beige Book takeaways, or 2026 outlook...")
if user_in:
    st.chat_message("user").write(user_in)
    st.session_state.messages.append({"role":"user","content":user_in})
    stream = run_query(user_in)
    final = st.chat_message("assistant",avatar="🤖").write_stream(stream)
    st.session_state.messages.append({"role":"assistant","content":final})
