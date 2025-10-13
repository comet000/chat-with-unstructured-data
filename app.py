import streamlit as st
import textwrap
from snowflake.snowpark import Session
from typing import List
import json

# Connection setup
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

# -- Page Title --
st.title("Chat with Cortex Search RAG")

# ------------------------------------------------------------------
# Helper Functions for Cortex (SQL-based, works on Streamlit Cloud)
# ------------------------------------------------------------------

def cortex_search_query(session, database, schema, service_name, query, columns=None, limit=5):
    """
    Query Cortex Search Service via SQL.
    Returns a list of result dictionaries.
    """
    columns_param = f", COLUMNS => ARRAY_CONSTRUCT({', '.join([f\"'{c}'\" for c in columns])})" if columns else ""
    query_escaped = query.replace("'", "''")
    
    sql = f"""
        SELECT PARSE_JSON(results) as results
        FROM TABLE(
            {database}.{schema}.{service_name}!SEARCH(
                QUERY => '{query_escaped}'
                {columns_param},
                LIMIT => {limit}
            )
        )
    """
    
    try:
        result = session.sql(sql).collect()
        if result:
            # Parse the JSON results
            results_list = []
            for row in result:
                results_list.append(json.loads(row['RESULTS']))
            return results_list
        return []
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def cortex_complete_stream(session, model, messages):
    """
    Stream completion from Cortex Complete via SQL.
    Yields chunks of text.
    """
    # Format messages as JSON
    messages_json = json.dumps(messages).replace("'", "''")
    
    sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            {messages_json}
        ) as response
    """
    
    try:
        result = session.sql(sql).collect()
        if result:
            response_text = result[0]['RESPONSE']
            # Simulate streaming by yielding the full response
            # (Cortex Complete doesn't support true streaming via SQL)
            yield response_text
    except Exception as e:
        yield f"Error generating response: {str(e)}"

# ------------------------------------------------------------------
# CortexSearchRetriever (SQL-based)
# ------------------------------------------------------------------
class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 2):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self._database = "CORTEX_SEARCH_TUTORIAL_DB"
        self._schema = "PUBLIC"
        self._service = "FOMC_SEARCH_SERVICE"

    def retrieve(self, query: str) -> List[str]:
        """Retrieve chunks using Cortex Search via SQL"""
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
            # Extract chunks from results
            chunks = []
            for result in results:
                if isinstance(result, dict) and "chunk" in result:
                    chunks.append(result["chunk"])
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "chunk" in item:
                            chunks.append(item["chunk"])
            return chunks
        return []

# ------------------------------------------------------------------
# RAG class
# ------------------------------------------------------------------
class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(session, limit_to_retrieve=5)

    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve relevant text from vector store"""
        return self.retriever.retrieve(query)

    def build_messages_with_context(self, conversation_messages, context_chunks):
        """
        Takes the entire conversation and appends a system message with context.
        """
        updated_messages = list(conversation_messages)

        if context_chunks:
            context_str = "\n\n".join([f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
            context_message_content = (
                f"You have retrieved the following context:\n\n"
                f"{context_str}\n\n"
                "Based on the conversation and the context above, please answer the last user question "
                "in a comprehensive and helpful way. If the context doesn't contain relevant information, "
                "acknowledge that and provide a general answer."
            )
        else:
            context_message_content = (
                "No specific context was retrieved. Please answer based on general knowledge, "
                "but acknowledge the lack of specific context."
            )
        
        updated_messages.append({"role": "system", "content": context_message_content})
        return updated_messages

    def generate_completion_stream(self, messages):
        """Stream the response using Cortex Complete"""
        return cortex_complete_stream(session, "claude-3-5-sonnet", messages)

# Instantiate the RAG
rag = RAG()

# ------------------------------------------------------------------
# Streamlit Chat Logic
# ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear Conversation"):
    st.session_state.messages.clear()
    st.rerun()

def display_messages():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.chat_message("user").write(content)
        elif role == "assistant":
            st.chat_message("assistant", avatar="🤖").write(content)

# Render existing messages
display_messages()

def answer_question_using_rag(query: str):
    """
    1) Retrieve context chunks
    2) Build message array with context
    3) Stream the LLM response
    """
    # Retrieve context
    with st.spinner("Retrieving context..."):
        context_chunks = rag.retrieve_context(query)

    # Show context
    if context_chunks:
        st.write("**Relevant Context Found:**")
        with st.expander("See retrieved context"):
            for i, chunk in enumerate(context_chunks):
                wrapped_chunk = textwrap.fill(chunk, width=80)
                st.info(f"**Context {i+1}:**\n{wrapped_chunk}")
    else:
        st.warning("No relevant context found. Answering from general knowledge.")

    # Build messages with context
    updated_messages = rag.build_messages_with_context(st.session_state.messages, context_chunks)

    # Stream the response
    with st.spinner("Generating response..."):
        stream = rag.generate_completion_stream(updated_messages)
    return stream

def main():
    user_input = st.chat_input("Ask your question about FOMC or economic data...")

    if user_input:
        # Append user message
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get RAG response
        stream = answer_question_using_rag(user_input)

        # Display streaming response
        final_text = st.chat_message("assistant", avatar="🤖").write_stream(stream)

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": final_text})

if __name__ == "__main__":
    main()
