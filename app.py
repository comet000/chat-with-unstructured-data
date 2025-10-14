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
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n{2,}|(?<=[.?!])\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]

def dedupe_context_texts(texts: List[str]) -> List[str]:
    seen, result = set(), []
    for t in texts:
        cleaned = re.sub(r"\s+", " ", t.strip().lower())
        if any(cleaned in s or s in cleaned for s in seen):
            continue
        seen.add(cleaned)
        result.append(t)
    return result

def extract_better_title(chunk: str) -> str:
    # Improved title extraction: Look for headers, dates, or key phrases
    lines = chunk.split('\n')
    if lines and re.match(r'^(#|\d{4}|\w+ \d{1,2}, \d{4}|Staff Economic Outlook|FEDERAL RESERVE press release|Voting for|CHAIR POWELL)', lines[0].strip()):
        return lines[0].strip()[:100]  # Use first line if it looks like a header or date
    paragraphs = split_paragraphs(chunk)
    if paragraphs and len(paragraphs[0].split()) < 15:  # Short first paragraph likely title
        return paragraphs[0].strip()
    # Fallback: Extract first sentence or key phrase
    first_sentence = re.split(r'(?<=[\.\!\?])\s', chunk.strip())[0]
    return first_sentence[:100] + '...' if len(first_sentence) > 100 else first_sentence

# --------------------------------------------------
# ✨ RAG Class
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)

    def retrieve_context(self, query: str) -> List[str]:
        chunks = self.retriever.retrieve(query)
        chunks = dedupe_context_texts(chunks)
        return chunks[:5]  # Limit to top 5 most useful (assuming retrieval is ranked)

    def summarize_context(self, contexts: List[str]) -> str:
        if not contexts:
            return "No relevant context retrieved."
        joined = "\n\n".join(contexts)
        prompt = (
            "You are an expert financial analyst familiar with FOMC policy statements, "
            "minutes, and economic outlooks. Summarize the following excerpts clearly and concisely, "
            "retaining key figures, policy stances, economic indicators, dates, and sources of expectations "
            "(e.g., staff projections, Chair statements). Organize by timeline or meeting date where possible:\n\n"
            f"{joined}"
        )
        summary = complete("claude-3-5-sonnet", prompt, session=session)
        return str(summary).strip()

    def build_messages_with_context(self, messages, context):
        summary = self.summarize_context(context)
        system_content = (
            "You are an expert economic analyst specializing in FOMC communications. "
            "Use the following summarized context from official FOMC documents to answer the user's question clearly and conversationally. "
            "In your response, specify what was expected or decided, when (e.g., specific meeting dates or projection years), "
            "by whom (e.g., FOMC staff, Chair Powell, the Committee), and why (e.g., based on economic indicators like job gains, inflation data). "
            "Explain policy statements and market implications in simple, accurate terms. "
            "If the answer cannot be found in the provided materials, politely say you don't have that information.\n\n"
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
            seen_titles = set()
            for chunk in chunks:
                cleaned = fix_text_formatting(chunk)
                paragraphs = split_paragraphs(cleaned)
                title = extract_better_title(cleaned)
                body = " ".join(paragraphs[1:]) if len(paragraphs) > 1 and len(paragraphs[0].split()) < 15 else " ".join(paragraphs)
                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())
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
