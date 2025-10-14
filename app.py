import streamlit as st
import re
from typing import List
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# ------------------------------------------------------------
# 🔧 SNOWFLAKE CONNECTION
# ------------------------------------------------------------
@st.cache_resource
def create_snowflake_session():
    connection_parameters = {
        "account": "fokiamm-yqb60913",
        "user": "streamlit_demo_user",
        "password": "RagCortex#78_Pw",
        "warehouse": "CORTEX_SEARCH_TUTORIAL_WH",
        "database": "CORTEX_SEARCH_TUTORIAL_DB",
        "schema": "PUBLIC",
    }
    return Session.builder.configs(connection_parameters).create()

session = create_snowflake_session()

# ------------------------------------------------------------
# 🎨 STYLING
# ------------------------------------------------------------
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

st.title("💬 FOMC Document Chat (Snowflake Cortex RAG)")

# ------------------------------------------------------------
# 🧠 UTILITIES
# ------------------------------------------------------------
def fix_text_formatting(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def dedupe_context_texts(texts: List[str]) -> List[str]:
    seen, result = set(), []
    for t in texts:
        key = t.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result

def extract_better_title(text: str) -> str:
    first_line = text.strip().split("\n")[0]
    return first_line[:90] + ("..." if len(first_line) > 90 else "")

STOPWORDS = {
    "the","and","for","with","this","that","from","have","been",
    "will","which","their","they","there","such","than","then",
    "into","when","what","where","while","about"
}

# ------------------------------------------------------------
# 🔍 RAG CLASS
# ------------------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session, limit_to_retrieve=10):
        self.session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str):
        # ✅ use Root API — not session.database()
        root = Root(self.session)
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
                .schemas["PUBLIC"]
                .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )

        # ✅ only query existing columns
        resp = search_service.search(query=query, columns=["chunk", "file_name"], limit=self._limit_to_retrieve)

        if resp.results:
            return [{"chunk": r["chunk"], "file_name": r.get("file_name", "Unknown_File")} for r in resp.results]
        return []

# ------------------------------------------------------------
# 🧩 RAG PIPELINE
# ------------------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(snowpark_session=session, limit_to_retrieve=10)

    def retrieve_context(self, query: str) -> List[str]:
        if "rag_cache" not in st.session_state:
            st.session_state.rag_cache = {}

        if query in st.session_state.rag_cache:
            return st.session_state.rag_cache[query]["chunks"]

        results = self.retriever.retrieve(query)
        chunks = [r["chunk"] for r in results]
        file_names = [r["file_name"] for r in results]
        chunks = dedupe_context_texts(chunks)

        st.session_state.rag_cache[query] = {"chunks": chunks[:5], "file_names": file_names[:5]}
        return chunks[:5]

rag = RAG()

# ------------------------------------------------------------
# 🤖 ANSWER GENERATION
# ------------------------------------------------------------
def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        chunks = rag.retrieve_context(query)

    st.markdown("### 🔍 Retrieved Context")
    if not chunks:
        st.warning("No relevant excerpts found.")
        return "No relevant excerpts found."

    cache_entry = st.session_state.rag_cache[query]
    file_names = cache_entry["file_names"]

    for i, chunk in enumerate(chunks):
        cleaned = fix_text_formatting(chunk)
        paragraphs = split_paragraphs(cleaned)
        title = extract_better_title(cleaned)
        body = " ".join(paragraphs)

        # 🔹 Highlight query words
        query_words = {w.lower() for w in query.split() if len(w) > 4 and w.lower() not in STOPWORDS}
        for word in query_words:
            body = re.sub(f"({re.escape(word)})", r"<mark>\\1</mark>", body, flags=re.IGNORECASE)

        source = file_names[i] if i < len(file_names) else "Unknown File"
        google_link = f"https://www.google.com/search?q={source.replace(' ', '+')}+filetype:pdf+FOMC"

        st.markdown(f"""
        <div class="context-card">
            <div class="context-title">{title}</div>
            <div class="context-body">{body[:700]}{'...' if len(body)>700 else ''}</div>
            <div style="margin-top:6px;"><a href="{google_link}" target="_blank">🔗 View Source: {source}</a></div>
        </div>
        """, unsafe_allow_html=True)

    # 🧠 Generate answer using Snowflake Cortex
    joined_context = "\n\n".join(chunks[:5])
    prompt = f"""You are an assistant summarizing FOMC policy context.
Use the excerpts below to answer accurately.

Context:
{joined_context}

Question: {query}

Answer:"""
    answer = complete("snowflake-arctic-instruct", prompt, session=session)
    return str(answer).strip()

# ------------------------------------------------------------
# 🚀 STREAMLIT UI
# ------------------------------------------------------------
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask a question about FOMC policy...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        answer = answer_question_using_rag(user_input)
        st.chat_message("assistant", avatar="🤖").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
