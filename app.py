import streamlit as st
import re
from typing import List, Dict, Tuple
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
.source-link:hover {text-decoration: underline;}
.debug-info {
    font-size: 11px;
    color: #666;
    margin-top: 4px;
    font-style: italic;
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
    test_query = session.sql("SELECT CURRENT_ROLE() as role, CURRENT_DATABASE() as database").collect()
    result = test_query[0]
    st.sidebar.success("✅ Connected to Snowflake")
    st.sidebar.write(f"👤 Role: {result['ROLE']}")
    st.sidebar.write(f"📊 Database: {result['DATABASE']}")
    
    try:
        root = Root(session)
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        test_resp = search_service.search(query="inflation", columns=["file_name"], limit=1)
        st.sidebar.success("✅ Cortex Search Service: WORKING")
    except Exception as e:
        st.sidebar.error(f"❌ Cortex Search Service: {str(e)}")
        
except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {e}")
    st.stop()

# --------------------------------------------------
# 🎯 Query Classifier - NEW
# --------------------------------------------------
class QueryClassifier:
    """Classify query intent to optimize retrieval strategy"""
    
    @staticmethod
    def classify(query: str) -> Dict[str, any]:
        query_lower = query.lower()
        
        classification = {
            "intent": "general",
            "temporal_preference": "balanced",
            "doc_type_preference": None,
            "target_year": None,
            "needs_summary": False,
            "needs_data": False
        }
        
        # Detect temporal intent
        if any(term in query_lower for term in ['most recent', 'latest', 'current', 'newest', 'last']):
            classification["temporal_preference"] = "recent"
        elif any(term in query_lower for term in ['historical', 'past', 'back in', 'during', 'was']):
            classification["temporal_preference"] = "historical"
        elif any(term in query_lower for term in ['outlook', 'forecast', 'projection', 'future', 'expect']):
            classification["temporal_preference"] = "recent"
            classification["intent"] = "forward_looking"
        
        # Detect specific year mentions
        year_match = re.search(r'20[2-3][0-9]', query)
        if year_match:
            classification["target_year"] = int(year_match.group())
            classification["temporal_preference"] = "specific_year"
        
        # Detect document type preference
        if 'beige book' in query_lower:
            classification["doc_type_preference"] = "beigebook"
        elif any(term in query_lower for term in ['minutes', 'meeting', 'fomc']):
            classification["doc_type_preference"] = "minutes"
        elif any(term in query_lower for term in ['projection', 'sep', 'forecast']):
            classification["doc_type_preference"] = "projection"
        elif 'press conference' in query_lower:
            classification["doc_type_preference"] = "presconf"
        
        # Detect if query wants summary
        if any(term in query_lower for term in ['takeaway', 'summary', 'main point', 'key point', 'highlight']):
            classification["needs_summary"] = True
        
        # Detect if query wants specific data
        if any(term in query_lower for term in ['rate', 'percent', 'inflation', 'unemployment', 'gdp', 'number']):
            classification["needs_data"] = True
        
        return classification

# --------------------------------------------------
# 🧠 Enhanced Retriever with Document Type Filtering
# --------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 15):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self._root = Root(snowpark_session)
        self.classifier = QueryClassifier()

    def retrieve(self, query: str) -> Tuple[List[dict], Dict]:
        try:
            # Classify the query first
            classification = self.classifier.classify(query)
            
            # Adjust retrieval limit based on needs
            retrieval_limit = self._limit_to_retrieve
            if classification["needs_summary"]:
                retrieval_limit = 20  # Get more for summary extraction
            
            search_service = (
                self._root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
                .schemas["PUBLIC"]
                .cortex_search_services["FOMC_SEARCH_SERVICE"]
            )
            
            # Enhanced query for better semantic matching
            enhanced_query = self._enhance_query(query, classification)
            
            resp = search_service.search(
                query=enhanced_query, 
                columns=["chunk", "file_name"], 
                limit=retrieval_limit
            )
            
            if resp.results:
                results = [{"chunk": r["chunk"], "file_name": r["file_name"]} for r in resp.results]
                
                # Apply document type filtering
                filtered_results = self._filter_by_doc_type(results, classification)
                
                # Apply intelligent sorting
                sorted_results = self._intelligent_sort(filtered_results, classification)
                
                # Deduplicate and ensure diversity
                final_results = self._ensure_diversity(sorted_results)
                
                return final_results[:10], classification  # Return top 10
            return [], classification
            
        except Exception as e:
            st.error(f"❌ Cortex Search Error: {e}")
            return [], classification

    def _enhance_query(self, query: str, classification: Dict) -> str:
        """Enhance query with additional context"""
        enhanced = query
        
        # Add document type hints
        if classification["doc_type_preference"] == "beigebook":
            enhanced += " beige book national summary economic conditions"
        elif classification["doc_type_preference"] == "projection":
            enhanced += " summary economic projections SEP forecast"
        
        # Add temporal hints
        if classification["target_year"]:
            enhanced += f" {classification['target_year']}"
        
        return enhanced

    def _filter_by_doc_type(self, results: List[dict], classification: Dict) -> List[dict]:
        """Filter results by document type preference"""
        doc_type = classification.get("doc_type_preference")
        if not doc_type:
            return results
        
        # Separate matching and non-matching
        matching = []
        others = []
        
        for r in results:
            file_name = r["file_name"].lower()
            if doc_type in file_name:
                matching.append(r)
            else:
                others.append(r)
        
        # Prioritize matching docs but keep some others for context
        return matching + others[:max(3, len(results) - len(matching))]

    def _intelligent_sort(self, results: List[dict], classification: Dict) -> List[dict]:
        """Sort based on classification"""
        temporal_pref = classification["temporal_preference"]
        target_year = classification["target_year"]
        
        if temporal_pref == "recent":
            return self._sort_by_date(results, reverse=True)
        elif temporal_pref == "historical":
            return self._sort_by_date(results, reverse=False)
        elif temporal_pref == "specific_year" and target_year:
            return self._sort_by_relevance_to_year(results, target_year)
        else:
            return self._balanced_sort(results)

    def _sort_by_date(self, results: List[dict], reverse: bool = True) -> List[dict]:
        """Sort by date extracted from filename"""
        def extract_date(file_name):
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
            if date_match:
                year, month, day = date_match.groups()
                return int(year + month + day)
            return 0
        
        return sorted(results, key=lambda x: extract_date(x["file_name"]), reverse=reverse)

    def _sort_by_relevance_to_year(self, results: List[dict], target_year: int) -> List[dict]:
        """Prioritize documents from target year"""
        def year_score(file_name):
            date_match = re.search(r'(\d{4})', file_name)
            if date_match:
                file_year = int(date_match.group(1))
                if file_year == target_year:
                    return 1000
                elif abs(file_year - target_year) == 1:
                    return 500
                else:
                    return 100
            return 0
        
        scored = [(year_score(r["file_name"]), i, r) for i, r in enumerate(results)]
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [r for _, _, r in scored]

    def _balanced_sort(self, results: List[dict]) -> List[dict]:
        """Balance recency with relevance"""
        def combined_score(idx, result):
            file_name = result["file_name"]
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file_name)
            
            score = 100 - idx  # Relevance score (search ranking)
            
            if date_match:
                year = int(date_match.group(1))
                if year == 2025:
                    score += 50
                elif year == 2024:
                    score += 30
                elif year == 2023:
                    score += 10
            
            return score
        
        scored = [(combined_score(i, r), r) for i, r in enumerate(results)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    def _ensure_diversity(self, results: List[dict]) -> List[dict]:
        """Ensure we don't return too many chunks from the same document"""
        seen_docs = {}
        diverse_results = []
        
        for result in results:
            file_name = result["file_name"]
            count = seen_docs.get(file_name, 0)
            
            # Allow max 3 chunks per document
            if count < 3:
                diverse_results.append(result)
                seen_docs[file_name] = count + 1
        
        return diverse_results

# --------------------------------------------------
# 🧹 Enhanced Utility Functions
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
    
    file_lower = file_name.lower()
    if 'minutes' in file_lower or 'fomcminutes' in file_lower:
        doc_type = "FOMC Minutes"
    elif 'proj' in file_lower or 'sep' in file_lower:
        doc_type = "Summary of Economic Projections"
    elif 'presconf' in file_lower:
        doc_type = "Press Conference"
    elif 'beigebook' in file_lower:
        doc_type = "Beige Book"
    elif 'statement' in file_lower:
        doc_type = "FOMC Statement"
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
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)
    cleaned = re.sub(r'\$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove common boilerplate
    boilerplate_phrases = [
        "What is the purpose of the Beige Book?",
        "How is the information in the Beige Book gathered?",
        "About This Publication"
    ]
    for phrase in boilerplate_phrases:
        if phrase in cleaned:
            cleaned = cleaned.replace(phrase, "")
    
    return cleaned.strip()

def is_substantive_content(chunk: str) -> bool:
    """Check if chunk contains substantive content vs boilerplate"""
    chunk_lower = chunk.lower()
    
    # Red flags for boilerplate
    boilerplate_indicators = [
        "what is the purpose",
        "table of contents",
        "about this publication",
        "how is the information",
        "federal reserve bank of"
    ]
    
    if any(indicator in chunk_lower for indicator in boilerplate_indicators):
        return False
    
    # Green flags for substantive content
    substantive_indicators = [
        "inflation",
        "employment",
        "economic",
        "growth",
        "percent",
        "rate",
        "policy",
        "outlook",
        "forecast",
        "projection"
    ]
    
    return any(indicator in chunk_lower for indicator in substantive_indicators)

# --------------------------------------------------
# ✨ Enhanced RAG Class
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)
        if "rag_cache" not in st.session_state:
            st.session_state.rag_cache = {}

    def retrieve_context(self, query: str) -> Tuple[List[dict], Dict]:
        cache_key = query.lower().strip()
        if cache_key in st.session_state.rag_cache:
            return (st.session_state.rag_cache[cache_key]["chunks"], 
                    st.session_state.rag_cache[cache_key]["classification"])
        
        chunks_with_metadata, classification = self.retriever.retrieve(query)
        
        # Filter out boilerplate
        substantive_chunks = [
            chunk for chunk in chunks_with_metadata 
            if is_substantive_content(chunk["chunk"])
        ]
        
        # If we filtered out everything, keep original
        if not substantive_chunks and chunks_with_metadata:
            substantive_chunks = chunks_with_metadata
        
        st.session_state.rag_cache[cache_key] = {
            "chunks": substantive_chunks,
            "classification": classification
        }
        return substantive_chunks, classification

    def summarize_context(self, contexts: List[dict], classification: Dict) -> str:
        if not contexts:
            return "No relevant context retrieved."
        
        # Take top 5 for summary
        limited_contexts = contexts[:5]
        chunk_texts = [item["chunk"] for item in limited_contexts]
        joined = "\n\n".join(chunk_texts)
        
        # Limit to prevent timeout
        if len(joined) > 6000:
            joined = joined[:6000] + "..."
        
        # Customize prompt based on classification
        if classification.get("needs_summary"):
            focus = "Focus on extracting the main takeaways and key points."
        elif classification.get("needs_data"):
            focus = "Focus on specific numerical data, rates, and projections."
        else:
            focus = "Provide a balanced summary of the key information."
        
        prompt = f"""
        You are an expert Federal Reserve analyst. {focus}
        
        Summarize the following excerpts clearly and concisely.
        Retain key figures, policy stances, economic indicators, and dates.
        Remove any boilerplate or procedural information.
        
        Context:
        {joined}
        
        Summary:
        """
        
        try:
            summary = complete("claude-3-5-sonnet", prompt, session=session)
            return str(summary).strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def build_messages_with_context(self, messages, context, classification):
        summary = self.summarize_context(context, classification)
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Build file list for verification
        file_list = list(set([item["file_name"] for item in context]))
        file_dates = []
        for f in file_list:
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', f)
            if date_match:
                year, month, day = date_match.groups()
                file_dates.append(f"{year}-{month}-{day}: {f}")
        
        system_content = f"""
        You are an expert Federal Reserve economic analyst.
        
        CRITICAL CONTEXT:
        - Today's date: {current_date}
        - All documents provided are real Federal Reserve publications
        - Documents available in context: {', '.join(file_dates)}
        
        INSTRUCTIONS:
        - Answer fully based on the provided context
        - If information isn't in the context, clearly state: "This information is not available in the retrieved documents"
        - When discussing specific meetings or reports, cite the date
        - For "most recent" queries, prioritize the latest dated document
        - Do NOT question document validity or dates
        - Provide specific numbers and data when available
        
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
    st.rerun()

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

def answer_question_using_rag(query: str):
    with st.spinner("Analyzing query and retrieving context..."):
        chunks_with_metadata, classification = rag.retrieve_context(query)

    with st.expander("🔍 See Retrieved Context & Query Analysis"):
        # Show query classification
        st.markdown("**Query Analysis:**")
        st.json(classification)
        
        st.markdown("---")
        st.markdown("**Retrieved Documents:**")
        
        if not chunks_with_metadata:
            st.info("No relevant context found via Cortex Search.")
        else:
            for idx, item in enumerate(chunks_with_metadata, 1):
                chunk = item["chunk"]
                file_name = item["file_name"]
                title = extract_clean_title(file_name)
                pdf_url = create_direct_link(file_name)
                
                cleaned_chunk = clean_chunk_content(chunk)
                
                st.markdown(f"""
                <div class="context-card">
                    <div class="context-title">#{idx} - {title}</div>
                    <div class="context-body">{cleaned_chunk[:600]}{'...' if len(cleaned_chunk)>600 else ''}</div>
                    <a href="{pdf_url}" target="_blank" class="source-link">
                        📄 View Full Document: {file_name}
                    </a>
                </div>
                """, unsafe_allow_html=True)

    if not chunks_with_metadata:
        return ["No relevant documents found. This may indicate the information is not available in the 2023-2025 FOMC document collection."]

    updated_messages = rag.build_messages_with_context(
        st.session_state.messages, 
        chunks_with_metadata,
        classification
    )
    
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
