import streamlit as st
import re
from typing import List
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# --------------------------------------------------
# 🧩 Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Chat with the Federal Reserve", page_icon="💬", layout="centered")
st.markdown("""
<style>
.stChatMessage {font-family: 'Inter', sans-serif; font-size: 15px;}
.stButton>button {border-radius: 8px; background-color: #0059ff; color:white;}
.stButton>button:hover {background-color:#0042cc;}
.context-card {
    border:1px solid #ddd; border-radius:10px; padding:12px;
    margin-bottom:12px; background-color:#f9f9f9;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}
.context-title {font-weight:600; font-size:15px; margin-bottom:8px;}
.context-body {font-size:13px; line-height:1.5; color:#333;}
.source-link {
    font-size: 12px;
    color: #0059ff;
    text-decoration: none;
    margin-top: 8px;
    display: inline-block;
}
.source-link:hover {
    text-decoration: underline;
}
/* Better table styling */
.stMarkdown table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}
.stMarkdown th, .stMarkdown td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
.stMarkdown th {
    background-color: #f2f2f2;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
st.title("💬 Chat with the Federal Reserve")
st.markdown("**RAG chat on all FOMC PDFs between 2023 - 2025**")

# --------------------------------------------------
# 🔑 Snowflake Session
# --------------------------------------------------
def create_snowflake_session():
    connection_parameters = {
        "account": st.secrets["account"],
        "user": st.secrets["user"],
        "password": st.secrets["password"],
        "warehouse": st.secrets["warehouse"],
        "database": st.secrets["database"],
        "schema": st.secrets["schema"],
        "role": st.secrets["role"],
    }
    return Session.builder.configs(connection_parameters).create()

try:
    session = create_snowflake_session()
    st.sidebar.success("✅ Connected to Snowflake")
except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {e}")
    st.stop()

# --------------------------------------------------
# 🔍 Check Available Search Services
# --------------------------------------------------
def list_search_services():
    """List all available Cortex Search Services"""
    try:
        root = Root(session)
        database = root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
        schema = database.schemas["PUBLIC"]
        services = list(schema.cortex_search_services)
        return services
    except Exception as e:
        st.error(f"Error listing search services: {e}")
        return []

# Display available search services in sidebar
with st.sidebar:
    st.subheader("Search Services")
    services = list_search_services()
    if services:
        st.write("Available services:")
        for service in services:
            st.write(f"- {service.name}")
    else:
        st.warning("No search services found")

# --------------------------------------------------
# 🧠 Retriever Class
# --------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 10):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str) -> List[dict]:
        try:
            root = Root(self._snowpark_session)
            search_service = (
                root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
                .schemas["PUBLIC"]
                .cortex_search_services["FOMC_SEARCH_SERVICE"]
            )
            resp = search_service.search(query=query, columns=["chunk", "file_name"], limit=self._limit_to_retrieve)
            if resp.results:
                return [{"chunk": r["chunk"], "file_name": r["file_name"]} for r in resp.results]
            return []
        except Exception as e:
            st.error(f"❌ Error retrieving data: {e}")
            st.info("""
            **Troubleshooting steps:**
            1. Check if the Cortex Search Service exists
            2. Verify your role has access to the service
            3. Ensure the service is in the CORTEX_SEARCH_TUTORIAL_DB.PUBLIC schema
            """)
            return []

# --------------------------------------------------
# 🧹 Utility Functions
# --------------------------------------------------
def fix_text_formatting(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    # Clean up math mode artifacts
    text = re.sub(r"\$", "", text)
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n{2,}|(?<=[.?!])\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]

def dedupe_context_texts(texts: List[dict]) -> List[dict]:
    seen, result = set(), []
    for item in texts:
        t = item["chunk"]
        cleaned = re.sub(r"\s+", " ", t.strip().lower())
        # Simple check for high overlap (e.g., if >80% substring match)
        if any(sum(1 for w in cleaned.split() if w in s.split()) / len(cleaned.split()) > 0.8 for s in seen):
            continue
        seen.add(cleaned)
        result.append(item)
    return result

def extract_clean_title(file_name: str) -> str:
    """Extract a clean title from the file name"""
    # Extract date from filename
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
    if date_match:
        year, month, day = date_match.groups()
        # Convert month number to month name
        month_names = {
            '01': 'January', '02': 'February', '03': 'March', '04': 'April',
            '05': 'May', '06': 'June', '07': 'July', '08': 'August',
            '09': 'September', '10': 'October', '11': 'November', '12': 'December'
        }
        month_name = month_names.get(month, month)
        formatted_date = f"{month_name} {int(day)}, {year}"
    else:
        formatted_date = "Unknown Date"
    
    # Determine document type
    if 'minutes' in file_name.lower():
        doc_type = "FOMC Minutes"
    elif 'proj' in file_name.lower() or 'sep' in file_name.lower():
        doc_type = "Summary of Economic Projections"
    elif 'presconf' in file_name.lower():
        doc_type = "Press Conference"
    elif 'transcript' in file_name.lower():
        doc_type = "Meeting Transcript"
    elif 'statement' in file_name.lower():
        doc_type = "Policy Statement"
    else:
        doc_type = "FOMC Document"
    
    return f"{doc_type} - {formatted_date}"

def create_direct_link(file_name: str) -> str:
    """Create direct link to Federal Reserve PDF with correct base URL"""
    # Different document types have different base URLs
    if 'presconf' in file_name.lower():
        base_url = "https://www.federalreserve.gov/mediacenter/files/"
    elif 'minutes' in file_name.lower():
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    elif 'proj' in file_name.lower() or 'sep' in file_name.lower():
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    elif 'statement' in file_name.lower():
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    elif 'transcript' in file_name.lower():
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    else:
        # Default to monetarypolicy for unknown types
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    
    return f"{base_url}{file_name}"

# --------------------------------------------------
# ✨ RAG Class
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)
        if "rag_cache" not in st.session_state:
            st.session_state.rag_cache = {}  # Cache for chunks and summaries per query

    def retrieve_context(self, query: str) -> List[dict]:
        if query in st.session_state.rag_cache:
            return st.session_state.rag_cache[query]["chunks"]
        chunks_with_metadata = self.retriever.retrieve(query)
        if not chunks_with_metadata:
            return []
        chunks_with_metadata = dedupe_context_texts(chunks_with_metadata)
        st.session_state.rag_cache[query] = {"chunks": chunks_with_metadata[:5]}
        return chunks_with_metadata[:5]

    def summarize_context(self, contexts: List[dict]) -> str:
        query = list(st.session_state.rag_cache.keys())[-1]  # Assume last query
        if "summary" in st.session_state.rag_cache.get(query, {}):
            return st.session_state.rag_cache[query]["summary"]
        if not contexts:
            return "No relevant context retrieved."
        
        # Extract just the chunk text for summarization
        chunk_texts = [item["chunk"] for item in contexts]
        joined = "\n\n".join(chunk_texts)
        
        prompt = (
            "You are an expert financial analyst familiar with FOMC policy statements, "
            "minutes, and economic outlooks. Summarize the following excerpts clearly and concisely, "
            "retaining key figures, policy stances, economic indicators, dates, and sources of expectations "
            "(e.g., staff projections vs. participant views, Chair statements). Include any dissenting views or risks. "
            "Flag any contradictions or evolutions across chunks (e.g., 'Projections revised up from Dec to March due to...'). "
            "Differentiate staff vs. Committee/Chair views. Organize by timeline, meeting date, or projection year where possible:\n\n"
            f"{joined}"
        )
        try:
            summary = complete("claude-3-5-sonnet", prompt, session=session)
            summary_str = str(summary).strip()
            st.session_state.rag_cache[query]["summary"] = summary_str
            return summary_str
        except Exception as e:
            return f"Error generating summary: {e}"

    def build_messages_with_context(self, messages, context):
        summary = self.summarize_context(context)
        system_content = (
            "You are an expert economic analyst specializing in FOMC communications. "
            "Use the following summarized context from official FOMC documents to answer the user's question clearly and conversationally. "
            "In your response, specify what was expected or decided, when (e.g., specific meeting dates or projection years), "
            "by whom (e.g., FOMC staff, Chair Powell, the Committee), and why (e.g., based on economic indicators like job gains, inflation data). "
            "Explain policy statements and market implications in simple, accurate terms. Structure responses with bullet points or numbered lists for clarity. "
            "IMPORTANT: When presenting data comparisons, use clean, simple markdown tables with clear headers. Avoid complex table structures."
            "Always cite specific dates/meetings and explain 'why' with evidence from indicators. "
            "If data is incomplete (e.g., no specifics for a year), explicitly state limitations and suggest why (e.g., 'Docs cover up to 2025, so 2026 is directional only'). "
            "For 'why' explanations, tie to multiple indicators if available (e.g., 'due to tariffs boosting import costs and persistent wage growth'). "
            "Use simple tables for any comparisons or progressions; if no table fits, use bullets with sub-bullets for 'why'. "
            "Add market implications where relevant (e.g., 'This could signal... for investors'). "
            "Keep responses concise—aim for 300-500 words max. End with a 1-2 sentence summary of implications. "
            "If the answer cannot be found in the provided materials, politely say you don't have that information; based on 2023-2025 docs.\n\n"
            f"Context Summary:\n{summary}"
        )
        updated = list(messages)
        updated.append({"role": "system", "content": system_content})
        return updated

    def generate_completion_stream(self, messages):
        try:
            return complete("claude-3-5-sonnet", messages, stream=True, session=session)
        except Exception as e:
            st.error(f"Error generating completion: {e}")
            return [f"Error: Unable to generate response. Please check the search service configuration."]

rag = RAG()

# --------------------------------------------------
# 💬 Streamlit Chat Logic
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()
    st.session_state.rag_cache = {}

def display_messages():
    for m in st.session_state.messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant", avatar="🤖").write(content)

display_messages()

# --------------------------------------------------
# ⚙️ Main RAG Interaction
# --------------------------------------------------
def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        chunks_with_metadata = rag.retrieve_context(query)

    # 💡 Display retrieved context
    with st.expander("🔍 See Retrieved Context"):
        if not chunks_with_metadata:
            st.info("No relevant context retrieved. This may be because:")
            st.info("1. The Cortex Search Service is not properly configured")
            st.info("2. Your role doesn't have access to the service")
            st.info("3. There are no documents matching your query")
        else:
            seen_titles = set()
            for item in chunks_with_metadata:
                chunk = item["chunk"]
                file_name = item["file_name"]
                cleaned = fix_text_formatting(chunk)
                
                # Get clean title from file name
                title = extract_clean_title(file_name)
                
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                
                # Clean up the context text - remove headings and focus on content
                paragraphs = split_paragraphs(cleaned)
                # Skip heading-like paragraphs and get to the actual content
                content_paragraphs = []
                for para in paragraphs:
                    # Skip paragraphs that are just headings or very short
                    if len(para.split()) > 10 and not para.startswith('#'):
                        content_paragraphs.append(para)
                
                # If we filtered too much, use the original paragraphs
                if not content_paragraphs:
                    content_paragraphs = paragraphs
                
                body = " ".join(content_paragraphs)
                
                # Create direct PDF link
                pdf_url = create_direct_link(file_name)
                
                st.markdown(
                    f"""
                    <div class="context-card">
                        <div class="context-title">{title}</div>
                        <div class="context-body">{body[:600]}{'...' if len(body)>600 else ''}</div>
                        <a href="{pdf_url}" target="_blank" class="source-link">
                            📄 View Full Document: {file_name}
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if not chunks_with_metadata:
        return ["I couldn't retrieve any relevant documents. Please check if the Cortex Search Service is properly configured."]

    updated_messages = rag.build_messages_with_context(st.session_state.messages, chunks_with_metadata)
    with st.spinner("Generating response..."):
        stream = rag.generate_completion_stream(updated_messages)
    return stream

# --------------------------------------------------
# 🚀 Main Chat Loop
# --------------------------------------------------
def main():
    user_input = st.chat_input("Ask about FOMC policy, inflation outlook, or meeting summaries...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        stream = answer_question_using_rag(user_input)
        final_text = st.chat_message("assistant", avatar="🤖").write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": final_text})

if __name__ == "__main__":
    main()
