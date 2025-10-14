import streamlit as st
import re
from typing import List, Tuple
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# --------------------------------------------------
# 🧩 Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Chat with Cortex Search RAG", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.stChatMessage {font-family: 'Inter', sans-serif; font-size: 15px;}
.stButton>button {border-radius: 8px; background-color: #0059ff; color:white;}
.stButton>button:hover {background-color:#0042cc;}
.context-card {
    border:1px solid #ddd; border-radius:10px; padding:12px; 
    margin-bottom:12px; background-color:#f9f9f9;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
    transition: background-color 0.3s ease;
}
.context-card:hover {
    background-color: #f0f7ff;
}
.context-title {font-weight:600; font-size:15px;}
.context-body {font-size:14px; line-height:1.6; color:#333; margin-top:6px;}
</style>
""", unsafe_allow_html=True)

st.title("💬 Chat with Cortex Search RAG")


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
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 20):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str) -> List[Tuple[str, str]]:
        root = Root(self._snowpark_session)
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )

        resp = search_service.search(
            query=query, columns=["file_name", "chunk"], limit=self._limit_to_retrieve
        )
        if resp.results:
            return [(r.get("file_name", "Unknown Document"), r["chunk"]) for r in resp.results]
        return []


# --------------------------------------------------
# 🧹 Utility Functions
# --------------------------------------------------
def fix_text_formatting(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n{2,}|(?<=[.?!])\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def dedupe_context_texts(chunks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen, result = set(), []
    for source, t in chunks:
        cleaned = re.sub(r"\s+", " ", t.strip().lower())
        if any(cleaned in s or s in cleaned for s in seen):
            continue
        seen.add(cleaned)
        result.append((source, t))
    return result


# --------------------------------------------------
# ✨ RAG Class
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)

    def retrieve_context(self, query: str) -> List[Tuple[str, str]]:
        chunks = self.retriever.retrieve(query)
        return dedupe_context_texts(chunks)

    def summarize_context(self, contexts: List[Tuple[str, str]]) -> str:
        if not contexts:
            return "No relevant context retrieved."
        joined = "\n\n".join(chunk for _, chunk in contexts)
        prompt = (
            "You are an assistant analyzing excerpts from official Federal Open Market Committee (FOMC) documents. "
            "Use only the information provided in these excerpts to answer questions or summarize content. "
            "If the context is insufficient or the answer is unclear, explicitly say so. "
            "Focus on providing precise, well-structured, and factual information.\n\n"
            f"{joined}"
        )
        summary = complete("claude-3-5-sonnet", prompt, session=session)
        return str(summary).strip()

    def build_messages_with_context(self, messages, context):
        summary = self.summarize_context(context)
        system_content = (
            f"You have retrieved the following summarized context:\n{summary}\n\n"
            "Use this information to answer the user's question as accurately as possible. "
            "Cite relevant excerpts or summarize them clearly, and if the context does not include an answer, say so."
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
    with st.spinner("Retrieving relevant excerpts..."):
        chunks = rag.retrieve_context(query)

    # 💡 Display retrieved context
    with st.expander("🔍 See Retrieved Context"):
        if not chunks:
            st.info("No relevant context retrieved.")
        else:
            seen_titles = set()
            for i, (source, chunk) in enumerate(chunks):
                cleaned = fix_text_formatting(chunk)
                paragraphs = split_paragraphs(cleaned)
                title = f"Excerpt {i+1} — {source}"

                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                body = " ".join(paragraphs)
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

    if chunks:
        unique_sources = {s for s, _ in chunks}
        st.caption(f"Sources: {', '.join(unique_sources)}")

    return stream


# --------------------------------------------------
# 🚀 Main Chat Loop
# --------------------------------------------------
def main():
    user_input = st.chat_input("Ask a question about FOMC documents (2023–2025)...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        stream = answer_question_using_rag(user_input)
        final_text = st.chat_message("assistant", avatar="🤖").write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": final_text})


if __name__ == "__main__":
    main()
