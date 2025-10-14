import streamlit as st
import re
from typing import List
from datetime import datetime
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
    st.sidebar.success("✅ Connected to Snowflake")
except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {e}")
    st.stop()

# --------------------------------------------------
# 🧠 Advanced Retriever with Smart Filtering
# --------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 15):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self._root = Root(snowpark_session)

    def retrieve(self, query: str) -> List[dict]:
        try:
            # Enhanced query understanding
            enhanced_query, query_type = self._enhance_query(query)
            
            search_service = (
                self._root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
                .schemas["PUBLIC"]
                .cortex_search_services["FOMC_SEARCH_SERVICE"]
            )
            
            # Get more results for better filtering
            resp = search_service.search(
                query=enhanced_query, 
                columns=["chunk", "file_name"], 
                limit=self._limit_to_retrieve * 3  # Get more for filtering
            )
            
            if resp.results:
                results = [{"chunk": r["chunk"], "file_name": r["file_name"]} for r in resp.results]
                
                # Apply aggressive filtering and ranking
                filtered_results = self._filter_low_quality_content(results)
                deduped_results = self._deduplicate_results(filtered_results)
                ranked_results = self._rank_results(deduped_results, query, query_type)
                
                return ranked_results[:self._limit_to_retrieve]
            return []
            
        except Exception as e:
            st.error(f"❌ Cortex Search Error: {e}")
            return []

    def _enhance_query(self, query: str):
        """Enhance query and detect query type"""
        query_lower = query.lower()
        enhanced_query = query
        
        # Detect query type for specialized handling
        if any(term in query_lower for term in ['most recent', 'latest', 'current', 'newest', 'last']):
            query_type = 'most_recent'
            enhanced_query += " 2025 2024"  # Boost recent years
        elif 'beige book' in query_lower:
            query_type = 'beige_book'
            enhanced_query += " National Summary economic conditions commentary"
        elif any(term in query_lower for term in ['outlook', 'forecast', 'projection', '2026', '2025']):
            query_type = 'economic_outlook'
            enhanced_query += " projections forecast economic outlook"
        elif any(term in query_lower for term in ['inflation', 'pce', 'cpi']):
            query_type = 'inflation'
            enhanced_query += " projections expectations prices"
        elif any(term in query_lower for term in ['minutes', 'fomc meeting']):
            query_type = 'minutes'
            # Extract year for specific meeting queries
            year_match = re.search(r'20[2-3][0-9]', query)
            if year_match:
                enhanced_query += f" {year_match.group()}"
        else:
            query_type = 'general'
            
        return enhanced_query, query_type

    def _filter_low_quality_content(self, results: List[dict]) -> List[dict]:
        """Aggressively filter out low-quality content"""
        high_quality_results = []
        
        for result in results:
            chunk = result["chunk"]
            if self._is_high_quality_content(chunk):
                high_quality_results.append(result)
        
        # If we filtered out everything, return original results but warn
        if not high_quality_results and results:
            st.sidebar.warning("⚠️ Low-quality content detected in results")
            return results[:5]  # Return limited low-quality results as fallback
            
        return high_quality_results

    def _is_high_quality_content(self, chunk: str) -> bool:
        """Identify high-quality economic content vs. metadata/boilerplate"""
        chunk_lower = chunk.lower()
        
        # Low-quality indicators (table of contents, metadata, boilerplate)
        low_quality_indicators = [
            'table of contents', '# contents', 'about this publication',
            'what is the purpose', 'how is the information', 
            'federal reserve bank of', 'page', 'section', 'img-', '![img',
            'contents about this', 'this document summarizes', 'outreach for the',
            'the beige book is intended', 'contacts outside the federal reserve'
        ]
        
        # High-quality indicators (actual economic content)
        high_quality_indicators = [
            'national summary', 'staff economic outlook', 'participants\' views',
            'economic conditions', 'inflation', 'employment', 'unemployment',
            'growth', 'gdp', 'projections', 'forecast', 'outlook', 'discussion',
            'real gdp', 'pce price inflation', 'labor market', 'wages',
            'spending', 'activity', 'prices', 'economic activity',
            'committee', 'participants', 'staff', 'indicators', 'data'
        ]
        
        # Check for low-quality content
        low_quality_count = sum(3 for indicator in low_quality_indicators if indicator in chunk_lower)
        
        # Check for high-quality content
        high_quality_count = sum(1 for indicator in high_quality_indicators if indicator in chunk_lower)
        
        # Require minimum chunk length and high-quality signals
        if len(chunk.strip()) < 100:  # Very short chunks are usually low quality
            return False
            
        return high_quality_count > low_quality_count and high_quality_count >= 2

    def _deduplicate_results(self, results: List[dict]) -> List[dict]:
        """Remove duplicate and near-duplicate results"""
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            # Create a normalized version for comparison
            chunk_normalized = re.sub(r'\s+', ' ', result["chunk"].lower().strip())
            chunk_hash = hash(chunk_normalized[:500])  # Hash first 500 chars
            
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                unique_results.append(result)
                
        return unique_results

    def _rank_results(self, results: List[dict], query: str, query_type: str) -> List[dict]:
        """Rank results based on multiple factors"""
        scored_results = []
        
        for result in results:
            score = 0
            chunk = result["chunk"]
            file_name = result["file_name"]
            
            # Content quality score
            score += self._content_quality_score(chunk)
            
            # Temporal relevance score
            score += self._temporal_score(file_name, query_type)
            
            # Query relevance score
            score += self._query_relevance_score(chunk, query, query_type)
            
            # Document type score
            score += self._document_type_score(file_name, chunk, query_type)
            
            scored_results.append((score, result))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results]

    def _content_quality_score(self, chunk: str) -> int:
        """Score based on content quality and importance"""
        chunk_lower = chunk.lower()
        score = 0
        
        # High-value sections
        if 'national summary' in chunk_lower:
            score += 25
        elif 'staff economic outlook' in chunk_lower:
            score += 20
        elif 'summary of economic projections' in chunk_lower:
            score += 20
        elif 'participants\' views' in chunk_lower:
            score += 15
            
        # Economic indicators
        economic_terms = {
            'inflation': 3, 'gdp': 3, 'unemployment': 3, 'growth': 2,
            'projections': 3, 'forecast': 2, 'economic conditions': 3,
            'labor market': 2, 'prices': 2, 'wages': 2, 'spending': 2
        }
        
        for term, points in economic_terms.items():
            if term in chunk_lower:
                score += points
                
        return score

    def _temporal_score(self, file_name: str, query_type: str) -> int:
        """Score based on temporal relevance"""
        date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
        if date_match:
            year, month, day = map(int, date_match.groups())
            
            # Strong recency boost for "most recent" queries
            if query_type == 'most_recent':
                if year == 2025:
                    return 30
                elif year == 2024:
                    return 10
                elif year == 2023:
                    return 0
                    
            # Moderate recency preference for other queries
            if year == 2025:
                return 15
            elif year == 2024:
                return 8
            elif year == 2023:
                return 2
                
        return 0

    def _query_relevance_score(self, chunk: str, query: str, query_type: str) -> int:
        """Score based on query term matching"""
        query_terms = set(term.lower() for term in query.split() if len(term) > 3)
        chunk_lower = chunk.lower()
        
        score = 0
        for term in query_terms:
            if term in chunk_lower:
                score += 3
                
        # Special handling for query types
        if query_type == 'beige_book' and 'beigebook' in chunk_lower:
            score += 10
        elif query_type == 'economic_outlook' and any(term in chunk_lower for term in ['projections', 'forecast', 'outlook']):
            score += 8
            
        return score

    def _document_type_score(self, file_name: str, chunk: str, query_type: str) -> int:
        """Score based on document type matching"""
        file_lower = file_name.lower()
        chunk_lower = chunk.lower()
        
        if query_type == 'beige_book' and 'beigebook' in file_lower:
            if 'national summary' in chunk_lower:
                return 20
            return 10
        elif query_type == 'economic_outlook' and ('proj' in file_lower or 'sep' in file_lower):
            return 15
        elif query_type == 'minutes' and 'minutes' in file_lower:
            return 10
            
        return 0

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
        
        # Use all context but limit total length
        chunk_texts = [item["chunk"] for item in contexts]
        joined = "\n\n".join(chunk_texts)
        
        if len(joined) > 6000:
            joined = joined[:6000] + "..."
        
        prompt = f"""
        You are an expert financial analyst familiar with FOMC policy statements, minutes, and economic outlooks.
        Summarize the following excerpts clearly and concisely, focusing on the most relevant information for answering the user's question.
        Retain key figures, policy stances, economic indicators, dates, and sources of expectations.
        
        Context:
        {joined}
        
        Summary:
        """
        
        try:
            summary = complete("claude-3-5-sonnet", prompt, session=session)
            return str(summary).strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def build_messages_with_context(self, messages, context):
        summary = self.summarize_context(context)
        current_date = datetime.now().strftime("%B %d, %Y")
        
        system_content = f"""
        You are an expert economic analyst specializing in FOMC communications.
        Current date context: It is currently {current_date}.
        All documents in the provided context are real and current.
        
        Answer the question in long-form, fully and completely, based EXCLUSIVELY on the context provided.
        Extract and present ALL relevant information from the context.
        Do not speculate about information that might be missing from the context.
        Do not question the validity or dates of any documents in the context.
        If the context contains the information needed to answer the question, provide a complete answer.
        
        Context Summary:
        {summary}
        """
        
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
                
                st.markdown(f"""
                <div class="context-card">
                    <div class="context-title">{title}</div>
                    <div class="context-body">{chunk[:600]}{'...' if len(chunk)>600 else ''}</div>
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
