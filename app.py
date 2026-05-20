import streamlit as st
import re
import logging
import time
import html
import json
from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from snowflake.snowpark import Session
from snowflake.core import Root
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    except (KeyError, TypeError) as e:
        logging.error(f"Failed to load secrets from Streamlit: {e}")
        st.error("Failed to load Snowflake credentials from secrets. Please check your Streamlit Cloud secrets configuration.")
        st.stop()

    try:
        session = Session.builder.configs(connection_parameters).create()
        logging.info("Snowflake session created successfully")
        return session
    except Exception as e:
        logging.error(f"Failed to create Snowflake session: {e}")
        st.error(f"Cannot connect to Snowflake: {e}. Please check credentials and try again.")
        raise

try:
    session = create_snowflake_session()
    root = Root(session)
    search_service = (
        root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
        .schemas["PUBLIC"]
        .cortex_search_services["FOMC_SEARCH_SERVICE"]
    )
except Exception as e:
    st.error("Failed to initialize Snowflake connection. Please check logs and secrets configuration.")
    st.stop()


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
    fname = file_name.lower()
    if "beigebook" in fname:
        doc_type = "Beige Book"
    elif "longerungoals" in fname:
        doc_type = "FOMC Longer-Run Goals"
    elif "presconf" in fname:
        doc_type = "Press Conference"
    elif "projtabl" in fname:
        doc_type = "Projection Tables"
    elif "mprfullreport" in fname or "mpr" in fname:
        doc_type = "Monetary Policy Report"
    elif "monetary" in fname:
        doc_type = "Monetary Document"
    elif "financial-stability-report" in fname or "financial" in fname:
        doc_type = "Financial Stability Report"
    elif "minutes" in fname:
        doc_type = "FOMC Minutes"
    else:
        doc_type = "FOMC Document"
    return f"{doc_type} - {date_str}"

def create_direct_link(file_name: str) -> str:
    try:
        base = "https://www.federalreserve.gov"
        name = file_name.split("/")[-1]
        mapping = [
            (r"beigebook", f"{base}/monetarypolicy/files/"),
            (r"fomc_longerungoals", f"{base}/monetarypolicy/files/"),
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
        return f"{base}/monetarypolicy/files/{name}"
    except Exception as e:
        logging.error(f"create_direct_link failed for {file_name}: {e}")
        return f"https://www.federalreserve.gov/monetarypolicy/files/{file_name.split('/')[-1]}"


class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit: int = 12):
        self._session = snowpark_session
        self._limit = limit

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        safe_query = query.replace("'", "''")
        config = {
            "query": safe_query,
            "columns": ["CHUNK", "FILE_NAME"],
            "limit": self._limit * 3
        }
        config_json = json.dumps(config).replace("'", "''")

        sql = f"""
            SELECT PARSE_JSON(
                SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                    'CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.FOMC_SEARCH_SERVICE',
                    '{config_json}'
                )
            )['results'] AS results
        """

        try:
            df = self._session.sql(sql).collect()
            if not df:
                return []

            raw_results = json.loads(df[0]['RESULTS'])
            unique_docs = {}
            for r in raw_results:
                file_name = r.get('FILE_NAME', '')
                if file_name and file_name not in unique_docs:
                    unique_docs[file_name] = {
                        'chunk': r.get('CHUNK', ''),
                        'file_name': file_name
                    }

            docs = list(unique_docs.values())
            target_years = extract_target_years(query)
            if target_years:
                lower_year = min(target_years) - 1
                upper_year = max(target_years)
                docs = [d for d in docs if lower_year <= extract_file_year(d['file_name']) <= upper_year]

            docs.sort(key=lambda d: extract_file_year(d['file_name']), reverse=True)
            return docs[:self._limit]

        except Exception as e:
            logging.error(f"Retrieval error: {e}")
            return []

rag_retriever = CortexSearchRetriever(session)


def extractive_answer(query: str, contexts: List[dict]) -> str:
    stop_words = {'the', 'a', 'an', 'in', 'is', 'was', 'how', 'to', 'of', 'and',
                  'for', 'on', 'with', 'that', 'it', 'are', 'be', 'this', 'at', 'by',
                  'do', 'did', 'does', 'what', 'when', 'where', 'which', 'who', 'will'}
    query_words = set(re.findall(r'\w+', query.lower())) - stop_words

    all_sentences = []
    for ctx in contexts:
        chunk = clean_chunk(ctx.get('chunk', ''))
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        for sent in sentences:
            if len(sent.strip()) >= 40:
                all_sentences.append((sent.strip(), ctx.get('file_name', '')))

    scored = []
    for sent, file_name in all_sentences:
        sent_words = set(re.findall(r'\w+', sent.lower()))
        overlap = len(query_words & sent_words)
        score = overlap / max(len(query_words), 1)
        scored.append((score, sent, file_name))

    scored.sort(key=lambda x: -x[0])

    seen = set()
    top = []
    for score, sent, file_name in scored:
        if score < 0.1:
            break
        normalized = re.sub(r'\s+', ' ', sent.lower()[:80])
        if normalized not in seen:
            seen.add(normalized)
            top.append((sent, file_name))
        if len(top) >= 8:
            break

    if not top:
        return "No relevant information found in the available Federal Reserve documents for this query."

    result_parts = []
    for sent, file_name in top:
        title = extract_clean_title(file_name)
        result_parts.append(f"• {sent}\n  *— {title}*")

    return "\n\n".join(result_parts)


@st.cache_data(ttl=3600, show_spinner=False)
def retrieve_cached(query: str) -> List[dict]:
    if not query:
        return []
    try:
        return rag_retriever.retrieve(query)
    except Exception as e:
        logging.error(f"Retrieval error: {e}")
        return []


def create_pdf(messages: List[dict]) -> BytesIO:
    buffer = BytesIO()

    now = datetime.now(ZoneInfo("America/New_York"))
    hour = now.strftime("%I").lstrip('0')
    am_pm = now.strftime("%p").lower()
    current_time = now.strftime(f"%B %d, %Y {hour}:%M {am_pm} EDT")

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Chat History - {current_time}", styles["Title"]))
    story.append(Spacer(1, 12))

    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = html.escape(msg['content']).replace('\n', '<br/>')

        p_text = f"<b>{role}:</b><br/>{content}"
        story.append(Paragraph(p_text, styles["Normal"]))
        story.append(Spacer(1, 12))

        if msg["role"] == "assistant" and msg.get("contexts"):
            story.append(Paragraph("<b>Sources Used in Response:</b>", styles["Heading3"]))
            story.append(Spacer(1, 6))
            for c in msg["contexts"]:
                title = extract_clean_title(c['file_name'])
                link = create_direct_link(c['file_name'])
                source_text = f"• <a href='{link}' color='blue'>{title}</a>"
                story.append(Paragraph(source_text, styles["Normal"]))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer


def run_query(user_query: str):
    with st.chat_message("assistant", avatar="⚙️"):
        progress_bar = st.progress(0, text="Retrieving context from Federal Reserve records...")

        contexts = retrieve_cached(user_query)
        progress_bar.progress(50, text="Extracting relevant information...")

        if not contexts:
            response_text = "No relevant documents found for this query. Please try rephrasing or check https://www.federalreserve.gov."
        else:
            response_text = extractive_answer(user_query, contexts)

        progress_bar.progress(100, text="Complete!")
        time.sleep(0.3)
        progress_bar.empty()

        st.markdown(response_text)

    top_contexts = contexts[:3] if contexts else []
    st.session_state.messages.append({"role": "assistant", "content": response_text, "contexts": top_contexts})

    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]


st.set_page_config(
    page_title="Chat with the Federal Reserve",
    page_icon="🏛️",
    layout="centered"
)

st.markdown(
    """
    <div style='display: inline-flex; flex-direction: column; align-items: flex-end;'>
        <h2 style='margin: 0;'>📈 Federal Reserve AI Research Assistant</h2>
        <div style='font-weight: bold; font-size: 18px;'>
            10,000 pages of Fed insights at your fingertips (2023 - 2026)
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        avatar = "🧑‍💻" if msg["role"] == "user" else "⚙️"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"], unsafe_allow_html=False)

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    top_contexts = st.session_state.messages[-1].get("contexts", [])
    with st.expander("🔍 View References", expanded=False):
        if not top_contexts:
            st.markdown("No relevant documents found. Check https://www.federalreserve.gov.")
        else:
            for c in top_contexts:
                title = extract_clean_title(c["file_name"])
                pdf_url = create_direct_link(c["file_name"])
                snippet = clean_chunk(c["chunk"])[:350] + ("..." if len(c["chunk"]) > 350 else "")
                st.markdown(f"**[{title}]({pdf_url})**")
                st.caption(snippet)
                st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat"):
            st.session_state.messages.clear()
            st.cache_data.clear()
            st.rerun()
    with col2:
        pdf_buffer = create_pdf(st.session_state.messages)
        st.download_button("💾 Download Research", pdf_buffer, "Research_Log.pdf", "application/pdf")

user_input = st.chat_input("Ask the Fed about policy, inflation, outlooks, insights, or history...")
if user_input:
    st.chat_message("user", avatar="🧑‍💻").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input, "contexts": []})
    run_query(user_input)
    st.rerun()

st.sidebar.markdown(
    """
    <h3 style='text-align: right;'>Example Questions</h3>
    """,
    unsafe_allow_html=True
)
example_questions = [
    "What are the biggest issues affecting the economy in 2026?",
    "What will be the long-term impact of AI and automation on productivity, wage growth, and the overall demand for labor?",
    "What are greatest risks to financial stability over the next 12–18 months, and how are you monitoring them?",
    "Are businesses still struggling with costs?",
    "What's the median rate projection for next year?",
    "What's the Fed's plan going forward?",
    "To what extent do tariff policy and trade disruptions factor into your inflation outlook and decision-making?",
    "When and how fast should the Fed cut rates (if at all)?",
    "How exposed is the financial system to a shift in sentiment or asset revaluation?",
    "Are supply chain issues still showing up regionally?",
    "How did the FOMC view the economic outlook in mid-2023?",
    "What were the key points discussed in the FOMC meeting in January 2023?",
    "How did the FOMC assess the labor market in mid-2024?",
    "What was the fed funds rate target range effective September 19, 2024?",
]
for question in example_questions:
    if st.sidebar.button(question, key=f"example_{question[:50]}"):
        st.chat_message("user", avatar="🧑‍💻").write(question)
        st.session_state.messages.append({"role": "user", "content": question, "contexts": []})
        run_query(question)
        st.rerun()
