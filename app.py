import streamlit as st
import re
from typing import List
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
}
.context-title {font-weight:600; font-size:15px; margin-bottom:6px;}
.context-meta {font-size:12px; color:#666; margin-bottom:6px;}
.context-body {font-size:13px; line-height:1.5; color:#333; margin-top:6px;}
.stChatMessage p { line-height: 1.6; margin-bottom: 0.8em; }
.stChatMessage strong { color: #0042cc; }
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
# 🧠 Retriever Class (UNCHANGED - safe)
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
        resp = search_service.search(
            query=query, columns=["chunk"], limit=self._limit_to_retrieve
        )
        if resp.results:
            return [r["chunk"] for r in resp.results]
        return []


# --------------------------------------------------
# 🧹 Utility Functions (UNCHANGED core logic)
# --------------------------------------------------
def fix_text_formatting(text: str) -> str:
    # Keep original safe formatting fixes
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)
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


# --------------------------------------------------
# ✨ RAG Class (only prompt wording improved - safe)
# --------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)

    def retrieve_context(self, query: str) -> List[str]:
        chunks = self.retriever.retrieve(query)
        chunks = dedupe_context_texts(chunks)
        return chunks

    def summarize_context(self, contexts: List[str]) -> str:
        if not contexts:
            return "No relevant context retrieved."
        joined = "\n\n".join(contexts)
        # Slightly stronger grounding language for the summarizer (still uses same complete API)
        prompt = (
            "Summarize the following retrieved FOMC-related text into a concise, factual summary. "
            "Only use information present in the text (do not infer or invent). "
            "When there are numeric forecasts, include them and label the year or wording if present.\n\n"
            f"{joined}"
        )
        summary = complete("claude-3-5-sonnet", prompt, session=session)
        return str(summary).strip()

    def build_messages_with_context(self, messages, context):
        # Improved system instruction: encourages structure, factual answers, and no hallucination.
        summary = self.summarize_context(context)
        system_content = (
            "You are an expert economic analyst specializing in Federal Reserve policy and FOMC materials. "
            "Answer clearly and factually using **only** the provided summarized context. "
            "If the requested information is not present in the context, say you do not know.\n\n"
            f"---\n\n"
            f"📘 **Summarized Context from FOMC & Related Sources:**\n{summary}\n\n"
            "When answering:\n"
            "- Start with a concise 1–2 sentence summary.\n"
            "- Then list key facts or numbers as bullet points (if any).\n"
            "- End with a short interpretation or implication **only if** supported by the context.\n"
        )
        updated = list(messages)
        updated.append({"role": "system", "content": system_content})
        return updated

    def generate_completion_stream(self, messages):
        # unchanged: stream response from the same model and API you used before
        return complete("claude-3-5-sonnet", messages, stream=True, session=session)

rag = RAG()


# --------------------------------------------------
# 🧩 Answer Post-Formatting (safe UI-only)
# --------------------------------------------------
def format_final_answer(text: str) -> str:
    """Improve readability while preserving content."""
    if not text:
        return text
    text = str(text).strip()

    # Insert paragraph breaks after sentences (helps readability)
    text = re.sub(r"(?<=[.?!])\s+(?=[A-Z])", "\n\n", text)

    # Break long colon-led lists into bullets
    text = re.sub(r"(?<=:)\s*(?=[A-Z0-9])", "\n• ", text)

    # Emphasize important macro terms
    highlight_terms = [
        "inflation", "interest rate", "unemployment",
        "monetary policy", "Federal Reserve", "FOMC", "GDP", "economic growth", "core PCE"
    ]
    for term in highlight_terms:
        pattern = re.compile(rf"\b({term})\b", re.IGNORECASE)
        text = pattern.sub(r"**\1**", text)

    # Trim excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# --------------------------------------------------
# 💬 Streamlit Chat Logic (keeps your behavior)
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
            # assistant messages in chat history are already formatted before being saved
            st.chat_message("assistant", avatar="🤖").markdown(content)

display_messages()


# --------------------------------------------------
# ⚙️ Main RAG Interaction (unchanged retrieval + safety)
# --------------------------------------------------
def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        chunks = rag.retrieve_context(query)

    # 💡 Display retrieved context nicely (UI-only improvements)
    with st.expander("🔍 See Retrieved Context"):
        if not chunks:
            st.info("No relevant context retrieved.")
        else:
            seen_titles = set()
            for i, chunk in enumerate(chunks):
                cleaned = fix_text_formatting(chunk)
                paragraphs = split_paragraphs(cleaned)

                # Heuristic: short first paragraph likely a title/header
                if paragraphs and len(paragraphs[0].split()) < 10 and paragraphs[0][0].isupper():
                    title = paragraphs[0].strip()
                    body = " ".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
                else:
                    title = f"Excerpt {i+1}"
                    body = " ".join(paragraphs)

                # Deduplicate by title text to avoid repeated cards
                if title and title.lower() in seen_titles:
                    continue
                if title:
                    seen_titles.add(title.lower())

                # Clean whitespace and reduce repeated fragments
                body = re.sub(r"\s+", " ", body).strip()

                # Highlight a few domain keywords
                highlight_terms = ["inflation", "interest rate", "unemployment", "policy", "Federal Reserve", "FOMC"]
                for term in highlight_terms:
                    pattern = re.compile(rf"\b({term})\b", re.IGNORECASE)
                    body = pattern.sub(r"<b>\1</b>", body)

                st.markdown(
                    f"""
                    <div class="context-card">
                        <div class="context-title">📄 {title}</div>
                        <div class="context-body">{body[:800]}{'...' if len(body)>800 else ''}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption("Showing deduplicated, highlighted excerpts (truncated).")

    # Build messages with the improved, grounded system instruction
    updated_messages = rag.build_messages_with_context(st.session_state.messages, chunks)

    with st.spinner("Generating response..."):
        stream = rag.generate_completion_stream(updated_messages)

    return stream


# --------------------------------------------------
# 🚀 Main Chat Loop (formatting applied safely - does not modify retrieval)
# --------------------------------------------------
def main():
    user_input = st.chat_input("Ask your question about FOMC or economic data...")
    if user_input:
        # Display user message immediately and save to session
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get the streaming response (unchanged streaming usage)
        stream = answer_question_using_rag(user_input)

        # This will display streaming output in the chat as before and return final string
        final_text = st.chat_message("assistant", avatar="🤖").write_stream(stream)

        # Format the final answer for readability (UI-only). Save formatted into session history.
        formatted_answer = format_final_answer(final_text) if isinstance(final_text, str) else str(final_text)
        # Show formatted version in a small expander for clarity (won't break streaming)
        with st.expander("📝 Formatted answer (readable)"):
            st.markdown(formatted_answer)

        # Save formatted answer to session messages (so history is readable)
        st.session_state.messages.append({"role": "assistant", "content": formatted_answer})


if __name__ == "__main__":
    main()
