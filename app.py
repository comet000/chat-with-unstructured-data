import streamlit as st
from snowflake.snowpark import Session
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens_connectors.snowflake import CortexSearchRetriever

# ----------------------------
# CONFIG
# ----------------------------
ACCOUNT = "fokiamm-yqb60913"
USER = "streamlit_demo_user"
PASSWORD = "RagCortex#78_Pw"
WAREHOUSE = "CORTEX_SEARCH_TUTORIAL_WH"
DATABASE = "CORTEX_SEARCH_TUTORIAL_DB"
SCHEMA = "PUBLIC"

# ----------------------------
# SNOWFLAKE CONNECTION
# ----------------------------
@st.cache_resource
def create_session():
    connection_parameters = {
        "account": ACCOUNT,
        "user": USER,
        "password": PASSWORD,
        "warehouse": WAREHOUSE,
        "database": DATABASE,
        "schema": SCHEMA,
    }
    return Session.builder.configs(connection_parameters).create()

session = create_session()

# ----------------------------
# RAG RETRIEVAL CLASS
# ----------------------------
class RAG:
    def __init__(self):
        # Uses your existing Cortex Search index from Snowflake
        self.retriever = CortexSearchRetriever(
            snowpark_session=session,
            limit_to_retrieve=10
        )

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={SpanAttributes.RETRIEVAL.QUERY_TEXT: "query"}
    )
    def retrieve(self, query: str):
        return self.retriever.retrieve(query)

    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes={SpanAttributes.LLM.REQUEST_TEXT: "query"}
    )
    def generate(self, query: str, context: str):
        prompt = f"""
        You are an expert assistant helping answer questions about Snowflake Cortex data.
        Use only the following context when answering.

        Context:
        {context}

        Question:
        {query}

        Provide a clear, concise, and factual answer:
        """
        return complete(model="mistral-large", prompt=prompt)

rag = RAG()

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Chat with Your Snowflake Data", layout="wide")

st.title("💬 Chat with Your Documents (Snowflake Cortex)")
st.caption("Ask questions about your ingested documents directly from your Snowflake database.")

query = st.text_input("Enter your question:", placeholder="e.g. What does the FOMC policy statement say about inflation?")
submit = st.button("Ask")

if submit and query:
    with st.spinner("Retrieving relevant information..."):
        results = rag.retrieve(query)

        if not results:
            st.error("No relevant documents found in Cortex Search.")
        else:
            context_text = "\n\n".join([r["content"] for r in results])
            with st.spinner("Generating answer using Snowflake Cortex..."):
                answer = rag.generate(query, context_text)
                st.markdown("### 💡 Answer")
                st.write(answer)

                with st.expander("View retrieved context"):
                    for idx, r in enumerate(results, 1):
                        st.markdown(f"**Source {idx}:**")
                        st.write(r["content"])

else:
    st.info("Enter a question above to get started.")
