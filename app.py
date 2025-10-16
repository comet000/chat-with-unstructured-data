import streamlit as st
import re
import logging
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

st.title("🏦 Chat with the Federal Reserve")
st.markdown("**Built on 5000 pages of Fed documents from 2023 - 2025**")

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

    if "minutes" in file_name.lower():
        doc_type = "FOMC Minutes"
    elif "proj" in file_name.lower():
        doc_type = "Summary of Economic Projections"
    elif "presconf" in file_name.lower():
        doc_type = "Press Conference"
    elif "beigebook" in file_name.lower():
        doc_type = "Beige Book"
    else:
        doc_type = "FOMC Document"

    return f"{doc_type} - {date_str}"


def create_direct_link(file_name: str) -> str:
    base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    if "presconf" in file_name.lower():
        base_url = "https://www.federalreserve.gov/mediacenter/files/"
    return f"{base_url}{file_name}"

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

            unique_docs = {r["file_name"]: r for r in raw_results}
            docs = list(unique_docs.values())

            # Filter by target years if specified
            target_years = extract_target_years(query)
            if target_years:
                lower, upper = min(target_years) - 1, max(target_years)
                docs = [
                    d for d in docs
                    if lower <= extract_file_year(d["file_name"]) <= upper
                ]

            # Sort and limit results
            docs = sorted(docs, key=lambda d: extract_file_year(d["file_name"]), reverse=True)
            docs = docs[:self.limit]

            return [{"chunk": d["chunk"], "file_name": d["file_name"]} for d in docs]

        except Exception as e:
            logging.error(f"Cortex Search retrieval error: {e}")
            st.error(f"❌ Cortex Search Error: {e}")
            return []


rag_retriever = CortexSearchRetriever(session)

# ======================================================
# 📘 PROMPT GENERATION
# ======================================================

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
        year_buckets.setdefault(year, []).append(clean_chunk(c["chunk"]))

    grouped_texts = [
        f"Year {year} excerpts:\n{chr(10).join(year_buckets[year])}"
        for year in sorted(year_buckets.keys())
    ]

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

Please answer fully and cite rel
