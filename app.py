import streamlit as st
import textwrap
from snowflake.snowpark import Session
from typing import List
import json

# ---------------------------------------------------------
# Connection Setup
# ---------------------------------------------------------
connection_parameters = {
    "account": st.secrets["account"],
    "user": st.secrets["user"],
    "password": st.secrets["password"],
    "warehouse": st.secrets["warehouse"],
    "database": st.secrets["database"],
    "schema": st.secrets["schema"],
    "role": st.secrets["role"]
}

session = Session.builder.configs(connection_parameters).create()

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("💬 Chat with Cortex Search RAG")

# ---------------------------------------------------------
# Cortex Search Helper (✅ fixed syntax)
# ---------------------------------------------------------
def cortex_search_query(session, database, schema, service_name, query, columns=None, limit=5):
    """
    Query Cortex Search Service via SQL.
    Returns a list of result dictionaries.
    """
    query_escaped = query.replace("'", "''")

    # Build column array
    if columns:
        columns_array = "ARRAY_CONSTRUCT(" + ", ".join([f"'{c}'" for c in columns]) + ")"
    else:
        columns_array = "NULL"

    # ✅ Correct syntax for Cortex Search
    # Must not include database/schema prefix before the service name
    sql = f"""
        USE SCHEMA {database}.{schema};
        SELECT * FROM TABLE(
            {service_name}!SEARCH(
                '{query_escaped}',
                {columns_array},
                {limit}
            )
        );
    """

    try:
        result = session.sql(sql).collect()
        if result:
            return [row.as_dict() for row in result]
        return []
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# ---------------------------------------------------------
# Cortex Complete Helper (for LLM responses)
# ---------------------------------------------------------
def cortex_complete(session, model, messages):
    """
    Call Snowflake Cortex Complete (non-streaming).
    """
    messages_json = json.dumps(messages).replace("'", "''")

    sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            PARSE_JSON('{messages_json}')
        ) AS RESPONSE
    """

    try:
        result = session.sql(sql).collect()
        if result:
            return result[0]['RESPONSE']
        return "No response generated."
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ---------------------------------------------------------
# Retriever Class
# ---------------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 3):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self._database = "CORTEX_SEARCH_TUTORIAL_DB"
        self._schema = "PUBLIC"
        self._service = "FOMC_SEARCH_SERVICE"

    def retrieve(self, query: str) -> List[str]:
        """Retrieve text chunks from Cortex Search"""
        results = cortex_search_query(
            self._snowpark_session,
            self._database,
            self._schema,
            self._service,
            query,
            columns=["chunk"],
            limit=self._limit_to_retrieve
        )
        if results:
            chunks = []
            for r in results:
                if "CHUNK" in r:
                    chunks.append(r["CHUNK"])
                elif "chunk" in r:
                    chunks.append(r["chunk"])
            return chunks
        return []

# ---------------------------------------------------------
# RAG Logic
# ---------------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session, limit_to_retrieve=5)

    def retrieve_context(self, query: str):
        return self.retriever.retrieve(query)

    def build_messages_with_context(self, conversation_messages, context_chunks):
        updated_messages = list(conversation_messages)

        if context_chunks:
            context_str = "\n\n".join(
                [f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
            )
            context_message = (
                f"You have retrieved the following context:\n\n"
                f"{context_str}\n\n"
                "Based on the conversation and context above, answer the user's last question clearly. "
                "If context is not relevant, acknowledge that and answer from general knowledge."
            )
        else:
            context_message = (
                "No specific context retrieved. Answer based on general knowledge, "
                "and acknowledge that context is missing."
            )

        updated_messages.append({"role": "system", "content": context_message})
        return updated_messages

    def generate_response(self, messages):
        """Non-streaming response generation"""
        return cortex_complete(session, "claude-3-5-sonnet", messages)

# ---------------------------------------------------------
# Streamlit Chat Logic
# ---------------------------------------------------------
rag = RAG()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🧹 Clear Conversation"):
    st.session_state.messages.clear()
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

def answer_question_using_rag(query: str):
    with st.spinner("Retrieving context..."):
        context_chunks = rag.retrieve_context(query)

    if context_chunks:
        st.write("**Relevant Context Found:**")
        with st.expander("📄 See retrieved context"):
            for i, chunk in enumerate(context_chunks):
                st.info(f"**Context {i+1}:**\n{textwrap.fill(chunk, 80)}")
    else:
        st.warning("⚠️ No relevant context found. Answering from general knowledge.")

    updated_messages = rag.build_messages_with_context(st.session_state.messages, context_chunks)

    with st.spinner("Generating response..."):
        response_text = rag.generate_response(updated_messages)
    return response_text

# ---------------------------------------------------------
# Main Chat Input
# ---------------------------------------------------------
def main():
    user_input = st.chat_input("Ask your question about FOMC or economic data...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        answer = answer_question_using_rag(user_input)

        st.chat_message("assistant", avatar="🤖").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
