import streamlit as st
import re
from typing import List
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete
import logging

# Initialize session state keys
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = {}

st.set_page_config(page_title="Chat with the Federal Reserve", page_icon="💬", layout="centered")
st.title("💬 Chat with the Federal Reserve - Enhanced Conversational Mode")
st.markdown("**Supports multi-document reasoning, trend analysis, and Fed jargon explanation.**")

# Hide Streamlit default menu and footer for cleaner UI
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Snowflake connection
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

def extract_target_years(query: str) -> List[int]:
    return [int(y) for y in re.findall(r'20\d{2}', query)]

def extract_file_year(file_name: str) -> int:
    m = re.search(r'(\d{4})', file_name)
    return int(m.group(1)) if m else 0

def clean_chunk(chunk: str) -> str:
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
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

class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit: int = 20):
        self._root = Root(snowpark_session)
        self._service = search_service
        self.limit = limit

    def retrieve(self, query: str) -> List[dict]:
        try:
            raw_results = self._service.search(query=query, columns=["chunk", "file_name"], limit=150).results
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

glossary = """
Glossary:
- Dot Plot: A chart showing each FOMC participant's forecast for the federal funds rate.
- Longer-run Inflation Expectations: Fed members' inflation expectations beyond the near-term future.
- Beige Book: A report summarizing economic conditions across Fed districts, published 8 times/year.
- Federal Funds Rate Target: The interest rate that the Fed targets for overnight lending between banks.
"""

def build_system_prompt(query: str, contexts: List[dict]) -> str:
    year_buckets = {}
    for c in contexts[:7]:
        year = extract_file_year(c["file_name"])
        if year not in year_buckets:
            year_buckets[year] = []
        year_buckets[year].append(clean_chunk(c["chunk"]))
    grouped_texts = []
    for year in sorted(year_buckets.keys()):
        grouped_texts.append(f"Year {year} excerpts:\n{chr(10).join(year_buckets[year])}")
    context_text = "\n\n".join(grouped_texts)
    if len(context_text) > 3500:
        context_text = context_text[:3500]

    examples = """
Examples of good answers:

Q: How has the Fed's tone on inflation changed from 2023 to now?
A: Based on documents from 2023 through 2025, the Fed shifted from serious inflation concerns to cautiously optimistic language as inflation moderated.

Q: Did Powell mention gas prices recently?
A: In recent press conferences, Powell acknowledged volatility in gas prices but noted their impact on overall inflation has been lessening.

Q: Is the Fed planning rate cuts in 2025?
A: Projections and dot plot data from late 2024 suggest some participants anticipate rate cuts in 2025, but the Fed emphasizes data-dependence.

Please answer fully and cite relevant document years when appropriate.
"""

    prompt = f"""
You are an expert economic analyst specializing in Federal Reserve communications.

Today is {datetime.now():%B %d, %Y}.

{glossary}

Use ONLY the following excerpts from FOMC minutes, press conferences, projections, and Beige Books to answer the user's question below. Do not invent facts or speculate.

{examples}

Context excerpts by year:

{context_text}

User Question:
{query}

Answer:
"""
    return prompt

def generate_response_stream(query: str, contexts: List[dict]):
    prompt = build_system_prompt(query, contexts)
    return complete("claude-3-5-sonnet", prompt, stream=True, session=session)

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()
    st.session_state.rag_cache.clear()

for msg in st.session_state.messages:
    if msg["role"] != "system":  # Hide system messages from user
        st.chat_message(msg["role"]).write(msg["content"])

def run_query(user_query: str):
    with st.spinner("Searching..."):
        contexts = rag_retriever.retrieve(user_query)

    if not contexts:
        st.info("No relevant context found.")
        return ["No relevant documents found."]

    st.session_state.messages.append({"role": "system", "content": build_system_prompt(user_query, contexts)})
    with st.spinner("Generating response..."):
        return generate_response_stream(user_query, contexts)

user_input = st.chat_input("Ask the Fed about policy, inflation, outlooks, or Beige Book insights...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    stream = run_query(user_input)
    final_answer = st.chat_message("assistant", avatar="🤖").write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
