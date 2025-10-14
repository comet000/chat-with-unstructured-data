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
mark {
    background-color: #dbeafe; /* soft pale blue highlight */
    padding: 0.1em 0.25em;
    border-radius: 3px;
}
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

    def retrieve(self, query: str) -> List[dict]:
        root = Root(self._snowpark_session)
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        resp = search_service.search(query=query, columns=["chunk", "document_title"], limit=self._limit_to_retrieve)
        if resp.results:
            return [{"chunk": r["chunk"], "title": r.get("document_title", "Unknown Source")} for r in resp.results]
        return []

# --------------------------------------------------
# 🧹 Utility Functions
# --------------------------------------------------
def fix_text_formatting(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\$", "", text)
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n{2,}|(?<=[.?!])\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]

def dedupe_context_texts(texts: List[str]) -> List[str]:
    seen, result = set(), []
    for t in texts:
        cleaned = re.sub(r"\s+", " ", t.strip().lower())
        if any(sum(1 for w in cleaned.split() if w in s.split()) / len(cleaned.split()) > 0.8 for s in seen):
            continue
        seen.add(cleaned)
        result.append(t)
    return result

def extract_better_title(chunk: str) -> str:
    cleaned = fix_text_formatting(chunk)
    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(-\d{1,2})?,\s+\d{4}', cleaned)
    meeting_type = re.search(r'(Staff Economic Outlook|CHAIR POWELL|Minutes of the Federal Open Market Committee|Summary of Economic Projections|Participants\' Views)', cleaned)
    suffix = ""
    if date_match:
        suffix += f" ({date_match.group(0)})"
    if meeting_type:
        suffix += f" - {meeting_type.group(0)}"
    if suffix:
        first_sentence = re.split(r'(?<=[.!?])\s', cleaned.strip())[0][:50]
        return f"{first_sentence}{suffix}"[:100]
    paragraphs = split_paragraphs(cleaned)
    if paragraphs and len(paragraphs[0].split()) < 15:
        return paragraphs[0].strip() + suffix
    title_prompt = f"Summarize this chunk's topic in 10 words, including any date or meeting: {cleaned[:200]}"
    title = str(complete("claude-3-5-sonnet", title_prompt, session=session)).strip()
    return title + suffix

STOPWORDS = {"the", "and", "for", "with", "from", "this", "that"}

# --------------------------------------------------
# ✨ RAG Class
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)
        if "rag_cache" not in st.session_state:
            st.session_state.rag_cache = {}

    def retrieve_context(self, query: str) -> List[str]:
        if query in st.session_state.rag_cache:
            return st.session_state.rag_cache[query]["chunks"]
        results = self.retriever.retrieve(query)
        chunks = [r["chunk"] for r in results]
        titles = [r["title"] for r in results]
        chunks = dedupe_context_texts(chunks)
        st.session_state.rag_cache[query] = {"chunks": chunks[:5], "titles": titles[:5]}
        return chunks[:5]

    def summarize_context(self, contexts: List[str]) -> str:
        query = list(st.session_state.rag_cache.keys())[-1]
        if "summary" in st.session_state.rag_cache.get(query, {}):
            return st.session_state.rag_cache[query]["summary"]
        if not contexts:
            return "No relevant context retrieved."
        joined = "\n\n".join(contexts)
        prompt = (
            "You are an expert financial analyst familiar with FOMC policy statements, "
            "minutes, and economic outlooks. Summarize the following excerpts clearly and concisely..."
            f"\n\n{joined}"
        )
        summary = complete("claude-3-5-sonnet", prompt, session=session)
        summary_str = str(summary).strip()
        st.session_state.rag_cache[query]["summary"] = summary_str
        return summary_str

    def build_messages_with_context(self, messages, context):
        summary = self.summarize_context(context)
        system_content = (
            "You are an expert economic analyst specializing in FOMC communications.\n\n"
            f"Context Summary:\n{summary}"
        )
        updated = list(messages)
        updated.append({"role": "system", "content": system_content})
        return updated

    def generate_completion_stream(self, messages):
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
        role, content = m["role"], m["content"]
        st.chat_message(role).write(content) if role == "user" else st.chat_message("assistant", avatar="🤖").write(content)

display_messages()

# --------------------------------------------------
# ⚙️ Main RAG Interaction
# --------------------------------------------------
def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        chunks = rag.retrieve_context(query)

    with st.expander("🔍 See Retrieved Context"):
        if not chunks:
            st.info("No relevant context retrieved.")
        else:
            cache_entry = st.session_state.rag_cache[list(st.session_state.rag_cache.keys())[-1]]
            titles = cache_entry.get("titles", [])
            seen_titles = set()

            for i, chunk in enumerate(chunks):
                cleaned = fix_text_formatting(chunk)
                paragraphs = split_paragraphs(cleaned)
                title = extract_better_title(cleaned)
                body = " ".join(paragraphs[1:]) if len(paragraphs) > 1 and len(paragraphs[0].split()) < 15 else " ".join(paragraphs)
                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())
                source_title = titles[i] if i < len(titles) else "Unknown Source"
                search_link = f"https://www.google.com/search?q={source_title.replace(' ', '+')}+filetype:pdf+FOMC"

                query_words = {w.lower() for w in query.split() if len(w) > 4 and w.lower() not in STOPWORDS}
                for word in query_words:
                    body = re.sub(f"({re.escape(word)})", r"<mark>\\1</mark>", body, flags=re.IGNORECASE)

                st.markdown(
                    f"""
                    <div class="context-card">
                        <div class="context-title">{title}</div>
                        <div class="context-body">{body[:700]}{'...' if len(body)>700 else ''}</div>
                        <div style="margin-top:6px;">
                            <a href="{search_link}" target="_blank">🔗 View Source: {source_title}</a>
                        </div>
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
