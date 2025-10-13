import streamlit as st
import textwrap
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import List
import numpy as np
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
        self.retriever = CortexSearchRetriever(session, limit_to_retrieve=5)

    def retrieve_context(self, query: str) -> List[str]:
        return self.retriever.retrieve(query)

    def build_messages_with_context(self, conversation_messages, context_chunks):
        updated_messages = list(conversation_messages)
        # Combine context chunks into one string message for system prompt
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
            session=session  # Explicitly pass Snowflake session here
        )
        return stream

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
        context_chunks = rag.retrieve_context(query)

    # Debug: Show raw context chunks
    st.write("DEBUG - Context chunks raw data:", context_chunks)

    st.write("**Relevant Context Found:**")
    with st.expander("See retrieved context"):
        for chunk in context_chunks:
            if chunk:
                clean_chunk = str(chunk).strip()
                wrapped_chunk = textwrap.fill(clean_chunk, width=60)
                st.markdown(f"``````")  # Key fix: use st.markdown here

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
