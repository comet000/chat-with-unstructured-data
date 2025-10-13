import streamlit as st
import re
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import List, Dict
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
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 2):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve

    # Now returns list of dict with chunk text and source doc metadata
    def retrieve(self, query: str) -> List[Dict[str, str]]:
        root = Root(self._snowpark_session)
        search_service = (
            root.databases["CORTEX_SEARCH_TUTORIAL_DB"]
               .schemas["PUBLIC"]
               .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        resp = search_service.search(
            query=query,
            columns=["chunk", "source_doc"],
            limit=self._limit_to_retrieve
        )
        if resp.results:
            return [
                {"chunk": r["chunk"], "source": r.get("source_doc", "Unknown Source")}
                for r in resp.results
            ]
        else:
            return []

class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session, limit_to_retrieve=5)

    def retrieve_context(self, query: str) -> List[Dict[str, str]]:
        return self.retriever.retrieve(query)

    def build_messages_with_context(self, conversation_messages, context_items):
        updated_messages = list(conversation_messages)
        combined_context = "\n\n".join(str(item["chunk"]) for item in context_items)
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

def fix_stuck_words(text):
    # Insert space between lowercase-uppercase letter transitions
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Insert space after punctuation if missing
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    # Optional: insert space before digits if stuck to letters
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
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

def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        context_items = rag.retrieve_context(query)

    st.write("DEBUG - Context chunks raw data:", context_items)

    st.write("**Relevant Context Found:**")
    with st.expander("See retrieved context"):
        for item in context_items:
            source = item.get("source", "Unknown Source")
            chunk_text = item.get("chunk", "")
            clean_chunk = "".join(c for c in str(chunk_text) if c.isprintable()).strip()
            fixed_chunk = fix_stuck_words(clean_chunk)
            # Replace newlines with spaces to avoid crushing text in HTML div
            html_friendly_chunk = fixed_chunk.replace('\n', ' ')

            st.markdown(f"**Source:** {source}")
            st.markdown(f"""
                <div style="
                    max-height: 400px;
                    overflow-y: auto;
                    border:1px solid #ccc;
                    padding: 10px;
                    white-space: normal;
                    font-size: 14px;
                ">
                {html_friendly_chunk}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

    updated_messages = rag.build_messages_with_context(st.session_state.messages, context_items)

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
