import streamlit as st
import re
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import List
from snowflake.cortex import complete

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

st.title("Chat with Cortex Search RAG")

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

class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session)

    def retrieve_context(self, query: str) -> List[str]:
        chunks = self.retriever.retrieve(query)
        # Deduplicate exact chunks ignoring whitespace and case
        seen = set()
        filtered = []
        for chunk in chunks:
            norm = re.sub(r"\s+", " ", chunk.lower()).strip()
            if norm not in seen:
                seen.add(norm)
                filtered.append(chunk)
        return filtered

    def build_messages_with_context(self, messages, context):
        updated = list(messages)
        combined_context = "\n\n".join(str(c) for c in context)
        system_content = (
            f"You have retrieved the following context (do not hallucinate beyond it):\n"
            f"{combined_context}\n\n"
            "Based on the conversation so far and the context above, provide a comprehensive answer."
        )
        updated.append({"role": "system", "content": system_content})
        return updated

    def generate_completion_stream(self, messages):
        return complete("claude-3-5-sonnet", messages, stream=True, session=session)

def fix_text_formatting(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)
    return text

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n{2,}|(?<=[.?!])\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]

rag = RAG()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear Conversation"):
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

def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        chunks = rag.retrieve_context(query)

    st.write("DEBUG - Context chunks raw data:", chunks)

    with st.expander("See retrieved context"):
        for i, chunk in enumerate(chunks):
            cleaned = "".join(c for c in str(chunk) if c.isprintable()).strip()
            cleaned = fix_text_formatting(cleaned)
            if i == 0:
                st.markdown(f"### Source (approximate document title): {cleaned}")
            else:
                paragraphs = split_paragraphs(cleaned)
                for para in paragraphs:
                    para = para.replace("\n", " ")
                    st.markdown(
                        f"""
                        <div style="
                            max-height:250px; overflow-y:auto; border:1px solid #ccc;
                            padding:10px; font-size:14px; margin-bottom:10px;">
                            {para}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("---")

    updated_messages = rag.build_messages_with_context(st.session_state.messages, chunks)

    with st.spinner("Generating response..."):
        stream = rag.generate_completion_stream(updated_messages)

    return stream

def main():
    user_input = st.chat_input("Ask your question about FOMC or economic data...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        stream = answer_question_using_rag(user_input)

        final_text = st.chat_message("assistant", avatar="🤖").write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": final_text})

if __name__ == "__main__":
    main()
