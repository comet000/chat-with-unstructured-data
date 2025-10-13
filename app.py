import streamlit as st
import re
from typing import List
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete

# --------------------------------------------------
# 🌐 Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Chat with FOMC Data", page_icon="📊", layout="centered")

st.markdown("""
<style>
.context-card {
  border:1px solid #ddd; border-radius:10px; padding:12px; 
  margin-bottom:12px; background-color:#f9f9f9;
  box-shadow:1px 1px 3px rgba(0,0,0,0.05);
}
.context-title {font-weight:600; font-size:15px;}
.context-meta {font-size:12px; color:#777; margin-top:2px;}
.context-body {font-size:13px; line-height:1.5; color:#333; margin-top:6px;}
</style>
""", unsafe_allow_html=True)

st.title("💬 Chat with FOMC Economic Data (RAG Demo)")

# --------------------------------------------------
# ❄️ Snowflake Session
# --------------------------------------------------
def create_snowflake_session():
    params = {
        "account": st.secrets["account"],
        "user": st.secrets["user"],
        "password": st.secrets["password"],
        "warehouse": st.secrets["warehouse"],
        "database": st.secrets["database"],
        "schema": st.secrets["schema"],
        "role": st.secrets["role"],
    }
    return Session.builder.configs(params).create()

session = create_snowflake_session()


# --------------------------------------------------
# 🔎 Cortex Retriever
# --------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 8):
        self.session = snowpark_session
        self.limit = limit_to_retrieve

    def retrieve(self, query: str):
        root = Root(self.session)
        svc = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        resp = svc.search(
            query=query,
            columns=["chunk", "source", "year"],  # <- include metadata
            limit=self.limit,
        )
        return resp.results or []


# --------------------------------------------------
# 🧠 RAG Pipeline
# --------------------------------------------------
def clean_text(t):
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t

def dedupe_texts(texts):
    seen, out = set(), []
    for t in texts:
        key = clean_text(t).lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)

    def retrieve_context(self, query):
        raw = self.retriever.retrieve(query)
        chunks = []
        for r in raw:
            chunk = clean_text(r.get("chunk"))
            meta = {
                "chunk": chunk,
                "source": r.get("source", "Unknown source"),
                "year": r.get("year", "—"),
            }
            chunks.append(meta)
        # dedupe
        unique = []
        seen = set()
        for c in chunks:
            norm = c["chunk"].lower()
            if norm not in seen:
                seen.add(norm)
                unique.append(c)
        return unique

    def rerank_context(self, query, contexts):
        # lightweight reranking using Claude
        prompt = f"""Given the query:
{query}

Rank the following text excerpts from most to least relevant. 
Return a JSON list of indexes (0-based) sorted by relevance.

Excerpts:
{[c["chunk"][:400] for c in contexts]}
"""
        resp = complete("claude-3-5-sonnet", prompt, session=session)
        try:
            order = [int(i) for i in re.findall(r"\d+", str(resp))]
            ordered = [contexts[i] for i in order if i < len(contexts)]
            return ordered[:6]
        except:
            return contexts[:6]

    def summarize_contexts(self, contexts):
        text_blocks = "\n\n".join(
            [f"[{c['source']} {c['year']}]\n{c['chunk']}" for c in contexts]
        )
        prompt = (
            "Summarize the following FOMC excerpts by year and topic. "
            "Highlight numeric forecasts and major policy remarks. "
            "Output should be concise, factual, and grouped by year:\n\n"
            f"{text_blocks}"
        )
        summary = complete("claude-3-5-sonnet", prompt, session=session)
        return str(summary).strip()

    def build_prompt(self, query, summary):
        system_prompt = (
            "You are an analyst assistant specializing in FOMC reports. "
            "Answer factually using only the summarized context provided below. "
            "Include numeric values and cite the year or source next to each number. "
            "If data for the requested period isn't present, state that clearly.\n\n"
            f"=== CONTEXT SUMMARY ===\n{summary}\n\n"
            f"=== QUESTION ===\n{query}\n"
        )
        return system_prompt

    def generate_stream(self, messages):
        return complete("claude-3-5-sonnet", messages, stream=True, session=session)


rag = RAG()


# --------------------------------------------------
# 💬 Chat Logic
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()

for m in st.session_state.messages:
    st.chat_message(m["role"], avatar="🤖" if m["role"] == "assistant" else None).write(m["content"])


def answer_with_rag(query):
    with st.spinner("🔎 Retrieving context..."):
        raw_contexts = rag.retrieve_context(query)
        contexts = rag.rerank_context(query, raw_contexts)

    with st.expander("📚 See Retrieved Context"):
        for c in contexts:
            st.markdown(
                f"""
                <div class="context-card">
                    <div class="context-title">{c['source']}</div>
                    <div class="context-meta">Year: {c['year']}</div>
                    <div class="context-body">{c['chunk'][:500]}{'...' if len(c['chunk'])>500 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.spinner("🧾 Summarizing context..."):
        summary = rag.summarize_contexts(contexts)

    messages = [{"role": "system", "content": rag.build_prompt(query, summary)}]

    with st.spinner("🤖 Generating response..."):
        stream = rag.generate_stream(messages)
    return stream


def main():
    query = st.chat_input("Ask about inflation, GDP, or FOMC policy...")
    if query:
        st.chat_message("user").write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        stream = answer_with_rag(query)
        final = st.chat_message("assistant", avatar="🤖").write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": final})


if __name__ == "__main__":
    main()
