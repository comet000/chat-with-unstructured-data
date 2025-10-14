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

try:
    session = create_snowflake_session()
    
    # Test basic connection
    test_query = session.sql("SELECT CURRENT_ROLE() as role, CURRENT_DATABASE() as database").collect()
    result = test_query[0]
    st.sidebar.success("✅ Connected to Snowflake")
    st.sidebar.write(f"👤 Role: {result['ROLE']}")
    st.sidebar.write(f"📊 Database: {result['DATABASE']}")
    
    # Test Cortex Search Service access
    try:
        root = Root(session)
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        
        # Quick test search
        test_resp = search_service.search(query="inflation", columns=["file_name"], limit=1)
        st.sidebar.success("✅ Cortex Search Service: WORKING")
        
    except Exception as e:
        st.sidebar.error(f"❌ Cortex Search Service: {str(e)}")
        
except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {e}")
    st.stop()

# --------------------------------------------------
# 🧠 Retriever Class - INTELLIGENT RANKING
# --------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 15):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self._root = Root(snowpark_session)

    def retrieve(self, query: str) -> List[dict]:
        try:
            search_service = (
                self._root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
                .schemas["PUBLIC"]
                .cortex_search_services["FOMC_SEARCH_SERVICE"]
            )
            
            # Get more results so we can filter and rank them properly
            resp = search_service.search(
                query=query, 
                columns=["chunk", "file_name"], 
                limit=self._limit_to_retrieve
            )
            
            if resp.results:
                results = [{"chunk": r["chunk"], "file_name": r["file_name"]} for r in resp.results]
                
                # Apply intelligent ranking and filtering
                ranked_results = self._rank_and_filter_results(results, query)
                return ranked_results
            return []
            
        except Exception as e:
            st.error(f"❌ Cortex Search Error: {e}")
            return []

    def _rank_and_filter_results(self, results: List[dict], query: str) -> List[dict]:
        """Intelligently rank and filter results for better relevance"""
        scored_results = []
        
        for result in results:
            score = 0
            chunk = result["chunk"]
            file_name = result["file_name"]
            
            # Score based on content quality (penalize table of contents, images)
            if self._is_high_quality_content(chunk):
                score += 10
            
            # Score based on recency
            score += self._get_recency_score(file_name)
            
            # Score based on query relevance
            score += self._get_query_relevance_score(chunk, query)
            
            # Score based on document type importance
            score += self._get_document_importance_score(file_name)
            
            scored_results.append((score, result))
        
        # Sort by score descending and take top 5
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results[:5]]

    def _is_high_quality_content(self, chunk: str) -> bool:
        """Filter out low-quality content like table of contents"""
        chunk_lower = chunk.lower()
        
        # Penalize table of contents, images, formatting
        low_quality_indicators = [
            'table of contents', 'contents', 'img-', '![img', '# contents',
            'federal reserve bank of', 'page', 'section'
        ]
        
        # Look for actual economic content
        high_quality_indicators = [
            'economic', 'growth', 'inflation', 'employment', 'outlook',
            'projections', 'forecast', 'expect', 'committee', 'participants',
            'staff', 'indicators', 'data', 'survey', 'conditions'
        ]
        
        low_quality_count = sum(1 for indicator in low_quality_indicators if indicator in chunk_lower)
        high_quality_count = sum(1 for indicator in high_quality_indicators if indicator in chunk_lower)
        
        return high_quality_count > low_quality_count

    def _get_recency_score(self, file_name: str) -> int:
        """Score based on how recent the document is"""
        date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
        if date_match:
            year, month, day = map(int, date_match.groups())
            # Higher score for more recent documents
            if year == 2025:
                return 20
            elif year == 2024:
                return 10
            elif year == 2023:
                return 5
        return 0

    def _get_query_relevance_score(self, chunk: str, query: str) -> int:
        """Score based on how relevant the chunk is to the query"""
        query_terms = set(term.lower() for term in query.split() if len(term) > 3)
        chunk_lower = chunk.lower()
        
        score = 0
        for term in query_terms:
            if term in chunk_lower:
                score += 5
                
        return score

    def _get_document_importance_score(self, file_name: str) -> int:
        """Score based on document type importance"""
        file_lower = file_name.lower()
        
        if 'minutes' in file_lower:
            return 8  # Meeting minutes are very important
        elif 'beigebook' in file_lower:
            return 7  # Beige Books are important economic snapshots
        elif 'proj' in file_lower or 'sep' in file_lower:
            return 9  # Economic projections are highly important
        elif 'presconf' in file_lower:
            return 6  # Press conferences are important
        else:
            return 5  # Default score for other documents

# --------------------------------------------------
# 🧹 Utility Functions
# --------------------------------------------------
def extract_clean_title(file_name: str) -> str:
    """Extract a clean title from the file name"""
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
    if date_match:
        year, month, day = date_match.groups()
        month_names = {
            '01': 'January', '02': 'February', '03': 'March', '04': 'April',
            '05': 'May', '06': 'June', '07': 'July', '08': 'August',
            '09': 'September', '10': 'October', '11': 'November', '12': 'December'
        }
        month_name = month_names.get(month, month)
        formatted_date = f"{month_name} {int(day)}, {year}"
    else:
        formatted_date = "Unknown Date"
    
    if 'minutes' in file_name.lower():
        doc_type = "FOMC Minutes"
    elif 'proj' in file_name.lower() or 'sep' in file_name.lower():
        doc_type = "Summary of Economic Projections"
    elif 'presconf' in file_name.lower():
        doc_type = "Press Conference"
    elif 'beigebook' in file_name.lower():
        doc_type = "Beige Book"
    else:
        doc_type = "FOMC Document"
    
    return f"{doc_type} - {formatted_date}"

def create_direct_link(file_name: str) -> str:
    """Create direct link to Federal Reserve PDF"""
    if 'presconf' in file_name.lower():
        base_url = "https://www.federalreserve.gov/mediacenter/files/"
    elif 'beigebook' in file_name.lower():
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    else:
        base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    return f"{base_url}{file_name}"

def clean_chunk_content(chunk: str) -> str:
    """Clean up chunk content for better display"""
    # Remove images and markdown formatting
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)  # Remove images
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)  # Remove markdown headers
    cleaned = re.sub(r'\$', '', cleaned)  # Remove math symbols
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    return cleaned.strip()

# --------------------------------------------------
# ✨ RAG Class
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)
        if "rag_cache" not in st.session_state:
            st.session_state.rag_cache = {}

    def retrieve_context(self, query: str) -> List[dict]:
        if query in st.session_state.rag_cache:
            return st.session_state.rag_cache[query]["chunks"]
        chunks_with_metadata = self.retriever.retrieve(query)
        st.session_state.rag_cache[query] = {"chunks": chunks_with_metadata}
        return chunks_with_metadata

    def summarize_context(self, contexts: List[dict]) -> str:
        if not contexts:
            return "No relevant context retrieved."
        
        chunk_texts = [item["chunk"] for item in contexts]
        joined = "\n\n".join(chunk_texts)
        
        prompt = (
            "You are an expert financial analyst familiar with FOMC policy statements, "
            "minutes, and economic outlooks. Summarize the following excerpts clearly and concisely, "
            "retaining key figures, policy stances, economic indicators, dates, and sources of expectations.\n\n"
            f"{joined}"
        )
        try:
            summary = complete("claude-3-5-sonnet", prompt, session=session)
            return str(summary).strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def build_messages_with_context(self, messages, context):
        summary = self.summarize_context(context)
        system_content = (
            "You are an expert economic analyst specializing in FOMC communications. "
            "Use the following summarized context from official FOMC documents to answer the user's question.\n\n"
            f"Context Summary:\n{summary}"
        )
        updated = list(messages)
        updated.append({"role": "system", "content": system_content})
        return updated

    def generate_completion_stream(self, messages):
        try:
            return complete("claude-3-5-sonnet", messages, stream=True, session=session)
        except Exception as e:
            return [f"Error: {e}"]

rag = RAG()

# --------------------------------------------------
# 💬 Streamlit Chat Logic
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()
    st.session_state.rag_cache = {}

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

def answer_question_using_rag(query: str):
    with st.spinner("Searching with Cortex Search..."):
        chunks_with_metadata = rag.retrieve_context(query)

    with st.expander("🔍 See Retrieved Context"):
        if not chunks_with_metadata:
            st.info("No relevant context found via Cortex Search.")
        else:
            for item in chunks_with_metadata:
                chunk = item["chunk"]
                file_name = item["file_name"]
                title = extract_clean_title(file_name)
                pdf_url = create_direct_link(file_name)
                
                # Clean up the chunk content
                cleaned_chunk = clean_chunk_content(chunk)
                
                st.markdown(f"""
                <div class="context-card">
                    <div class="context-title">{title}</div>
                    <div class="context-body">{cleaned_chunk[:600]}{'...' if len(cleaned_chunk)>600 else ''}</div>
                    <a href="{pdf_url}" target="_blank" class="source-link">
                        📄 View Full Document: {file_name}
                    </a>
                </div>
                """, unsafe_allow_html=True)

    if not chunks_with_metadata:
        return ["No relevant documents found via Cortex Search."]

    updated_messages = rag.build_messages_with_context(st.session_state.messages, chunks_with_metadata)
    with st.spinner("Generating response..."):
        stream = rag.generate_completion_stream(updated_messages)
    return stream

user_input = st.chat_input("Ask about FOMC policy, inflation outlook, or meeting summaries...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    stream = answer_question_using_rag(user_input)
    final_text = st.chat_message("assistant", avatar="🤖").write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": final_text})
