import streamlit as st
import re
from typing import List
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# --------------------------------------------------
# 🧩 Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Chat with FOMC Documents", page_icon="💬", layout="centered")
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
.context-title {font-weight:600; font-size:15px;}
.context-body {font-size:13px; line-height:1.5; color:#333; margin-top:6px;}
</style>
""", unsafe_allow_html=True)
st.title("💬 Chat with FOMC and Economic Policy Documents")

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
session = create_snowflake_session()

# --------------------------------------------------
# 🧠 Retriever Class
# --------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 10):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str) -> List[str]:
        root = Root(self._snowpark_session)
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        resp = search_service.search(query=query, columns=["chunk"], limit=self._limit_to_retrieve)
        if resp.results:
            return [r["chunk"] for r in resp.results]
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

def dedupe_context_texts(texts: List[str]) -> List[str]:
    seen, result = set(), []
    for t in texts:
        cleaned = re.sub(r"\s+", " ", t.strip().lower())
        # Simple check for high overlap (e.g., if >80% substring match)
        if any(sum(1 for w in cleaned.split() if w in s.split()) / len(cleaned.split()) > 0.8 for s in seen):
            continue
        seen.add(cleaned)
        result.append(t)
    return result

def extract_better_title(chunk: str) -> str:
    cleaned = fix_text_formatting(chunk)
    # Look for date patterns
    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(-\d{1,2})?,\s+\d{4}', cleaned)
    meeting_type = re.search(r'(Staff Economic Outlook|CHAIR POWELL|Minutes of the Federal Open Market Committee|Summary of Economic Projections|Participants\' Views)', cleaned)
    suffix = ""
    if date_match:
        suffix += f" ({date_match.group(0)})"
    if meeting_type:
        suffix += f" - {meeting_type.group(0)}"
    if suffix:
        first_sentence = re.split(r'(?<=[\.\!\?])\s', cleaned.strip())[0][:50]
        return f"{first_sentence}{suffix}"[:100]
    
    lines = cleaned.split('\n')
    if lines and re.match(r'^(#|\d{4}|\w+ \d{1,2}, \d{4}|Staff Economic Outlook|FEDERAL RESERVE press release|Voting for|CHAIR POWELL|Minutes of the Federal Open Market Committee|Summary of Economic Projections|Participants\' Views)', lines[0].strip()):
        return lines[0].strip()[:100] + suffix
    
    paragraphs = split_paragraphs(cleaned)
    if paragraphs and len(paragraphs[0].split()) < 15:
        return paragraphs[0].strip() + suffix
    
    # Fallback: Cortex summary title
    title_prompt = f"Summarize this chunk's topic in 10 words, including any date or meeting: {cleaned[:200]}"
    title = str(complete("claude-3-5-sonnet", title_prompt, session=session)).strip()
    return title + suffix

# Stopwords for highlighting
STOPWORDS = {"the", "and", "for", "with", "from", "this", "that"}

# --------------------------------------------------
# ✨ RAG Class
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)
        if "rag_cache" not in st.session_state:
            st.session_state.rag_cache = {}  # Cache for chunks and summaries per query

    def retrieve_context(self, query: str) -> List[str]:
        if query in st.session_state.rag_cache:
            return st.session_state.rag_cache[query]["chunks"]
        chunks = self.retriever.retrieve(query)
        chunks = dedupe_context_texts(chunks)
        st.session_state.rag_cache[query] = {"chunks": chunks[:5]}
        return chunks[:5]

    def summarize_context(self, contexts: List[str]) -> str:
        query = list(st.session_state.rag_cache.keys())[-1]  # Assume last query
        if "summary" in st.session_state.rag_cache.get(query, {}):
            return st.session_state.rag_cache[query]["summary"]
        if not contexts:
            return "No relevant context retrieved."
        joined = "\n\n".join(contexts)
        prompt = (
            "You are an expert financial analyst familiar with FOMC policy statements, "
            "minutes, and economic outlooks. Summarize the following excerpts clearly and concisely, "
            "retaining key figures, policy stances, economic indicators, dates, and sources of expectations "
            "(e.g., staff projections vs. participant views, Chair statements). Include any dissenting views or risks. "
            "Flag any contradictions or evolutions across chunks (e.g., 'Projections revised up from Dec to March due to...'). "
            "Differentiate staff vs. Committee/Chair views. Organize by timeline, meeting date, or projection year where possible:\n\n"
            f"{joined}"
        )
        summary = complete("claude-3-5-sonnet", prompt, session=session)
        summary_str = str(summary).strip()
        st.session_state.rag_cache[query]["summary"] = summary_str
        return summary_str

    def build_messages_with_context(self, messages, context):
        summary = self.summarize_context(context)
        system_content = (
            "You are an expert economic analyst specializing in FOMC communications. "
            "Use the following summarized context from official FOMC documents to answer the user's question clearly and conversationally. "
            "In your response, specify what was expected or decided, when (e.g., specific meeting dates or projection years), "
            "by whom (e.g., FOMC staff, Chair Powell, the Committee), and why (e.g., based on economic indicators like job gains, inflation data). "
            "Explain policy statements and market implications in simple, accurate terms. Structure responses with bullet points or numbered lists for clarity. "
            "If comparing figures across time or sources, present in a markdown table. Always cite specific dates/meetings and explain 'why' with evidence from indicators. "
            "If data is incomplete (e.g., no specifics for a year), explicitly state limitations and suggest why (e.g., 'Docs cover up to 2025, so 2026 is directional only'). "
            "For 'why' explanations, tie to multiple indicators if available (e.g., 'due to tariffs boosting import costs and persistent wage growth'). "
            "Use tables for any comparisons or progressions; if no table fits, use bullets with sub-bullets for 'why'. "
            "Add market implications where relevant (e.g., 'This could signal... for investors'). "
            "Keep responses concise—aim for 300-500 words max. End with a 1-2 sentence summary of implications. "
            "If the answer cannot be found in the provided materials, politely say you don't have that information; based on 2023-2025 docs.\n\n"
            f"Context Summary:\n{summary}"
        )
        updated = list(messages)
        updated.append({"role": "system", "content": system_content})
        return updated

    def generate_completion_stream(self, messages):
        # Optional: Internal verification (commented for perf; enable if needed)
        # verify_prompt = f"Verify this response draft against context summary: No hallucinations; cite only from context. Draft: {messages[-1]['content']}"
        # verified = complete("claude-3-5-sonnet", verify_prompt, session=session)
        return complete("claude-3-5-sonnet", messages, stream=True, session=session)

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
        chunks = rag.retrieve_context(query)

    # 💡 Display retrieved context
    with st.expander("🔍 See Retrieved Context"):
        if not chunks:
            st.info("No relevant context retrieved.")
        else:
            st.info("Showing top 5 relevant excerpts from FOMC documents (ranked by similarity). Click to expand for sources.")
            seen_titles = set()
            for chunk in chunks:
                cleaned = fix_text_formatting(chunk)
                paragraphs = split_paragraphs(cleaned)
                title = extract_better_title(cleaned)
                body = " ".join(paragraphs[1:]) if len(paragraphs) > 1 and len(paragraphs[0].split()) < 15 else " ".join(paragraphs)
                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())
                # Smarter highlighting: Filter query terms
                query_words = {w.lower() for w in query.split() if len(w) > 4 and w.lower() not in STOPWORDS}
                for word in query_words:
                    body = re.sub(f"({re.escape(word)})", r"<mark>\1</mark>", body, flags=re.IGNORECASE)
                st.markdown(
                    f"""
                    <div class="context-card">
                        <div class="context-title">{title}</div>
                        <div class="context-body">{body[:700]}{'...' if len(body)>700 else ''}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    updated_messages = rag.build_messages_with_context(st.session_state.messages, chunks)
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
