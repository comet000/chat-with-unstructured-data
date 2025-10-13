import streamlit as st
import re
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import List
from snowflake.cortex import complete

# Create Snowflake session explicitly with connection info from Streamlit secrets
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
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 5):
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
            query=query,
            columns=["chunk"],
            limit=self._limit_to_retrieve
        )
        if resp.results:
            return [r["chunk"] for r in resp.results]
        else:
            return []

class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session, limit_to_retrieve=10)

    def retrieve_context(self, query: str) -> List[str]:
        chunks = self.retriever.retrieve(query)

        # Simple duplication filtering by exact text content
        seen = set()
        filtered_chunks = []
        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk.strip().lower())
            if normalized not in seen:
                seen.add(normalized)
                filtered_chunks.append(chunk)
        return filtered_chunks

    def build_messages_with_context(self, conversation_messages, context_chunks):
        updated_messages = list(conversation_messages)
        combined_context = "\n\n".join(str(chunk) for chunk in context_chunks)
        context_message_content = (
            f"You have retrieved the following context (do not hallucinate beyond it):\n"
            f"{combined_context}\n\n"
            "Based on the conversation so far and the context above, please answer the last user question "
            "in a comprehensive, correct, and helpful way. If you don't have the information, just say so."
        )
        updated_messages.append({"role": "system", "content": context_message_content})
        return updated_messages

    def generate_completion_stream(self, messages):
        stream = complete(
            "claude-3-5-sonnet",
            messages,
            stream=True,
            session=session
        )
        return stream

def fix_stuck_words(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    return text

rag = RAG()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear Conversation"):
    st.session_state.messages.clear()

def display_messages():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.chat_message("user").write(content)
        elif role == "assistant":
            st.chat_message("assistant", avatar="🤖").write(content)
        elif role == "system":
            st.chat_message("assistant", avatar="🔧").write(f"[SYSTEM] {content}")

display_messages()

def split_into_paragraphs(text: str) -> List[str]:
    # Split text by 2+ newlines or punctuation + newline patterns to get paragraphs
    paragraphs = re.split(r'\n{2,}|(?<=[.?!])\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        context_chunks = rag.retrieve_context(query)

    st.write("DEBUG - Context chunks raw data:", context_chunks)

    st.write("**Relevant Context Found:**")
    with st.expander("See retrieved context"):
        for idx, chunk in enumerate(context_chunks):
            if chunk:
                clean_chunk = "".join(c for c in str(chunk) if c.isprintable()).strip()
                fixed_chunk = fix_stuck_words(clean_chunk)
                # Replace newlines with space for natural flow
                flow_chunk = fixed_chunk.replace('\n', ' ')

                if idx == 0:
                    st.markdown(f"### Source (document title): {flow_chunk}")
                else:
                    paragraphs = split_into_paragraphs(fixed_chunk)
                    for para in paragraphs:
                        flow_para = para.replace("\n", " ")
                        st.markdown(f"""
                            <div style="
                                max-height: 250px;
                                overflow-y: auto;
                                border:1px solid #ccc;
                                padding: 10px;
                                white-space: normal;
                                font-size: 14px;
                                margin-bottom: 10px;
                            ">
                            {flow_para}
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown("---")

    updated_messages = rag.build_messages_with_context(st.session_state.messages, context_chunks)

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
