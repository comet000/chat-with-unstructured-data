import streamlit as st
from snowflake.cortex import complete
from snowflake.cortex import SearchService

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="FOMC RAG Chatbot", layout="wide")

SERVICE_NAME = "FOMC_SEARCH_SERVICE"
MODEL_NAME = "mistral-large"  # or llama-3-70b if you prefer
NUM_RESULTS = 8
SIMILARITY_THRESHOLD = 0.35  # lower = looser retrieval, higher = stricter

# ----------------------------
# INITIALIZE SEARCH SERVICE
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_search_service():
    return SearchService(SERVICE_NAME)

svc = get_search_service()

# ----------------------------
# FUNCTIONS
# ----------------------------

def retrieve_context(query: str):
    """Retrieve most relevant excerpts from Cortex Search."""
    try:
        resp = svc.search(
            query=query,
            columns=["chunk", "source"],  # removed 'year' to fix 400 error
            limit=NUM_RESULTS,
        )
        results = [
            r for r in resp.results
            if r.similarity > SIMILARITY_THRESHOLD
        ]
        return results
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return []


def generate_answer(query: str, context: str):
    """Generate a concise, factual answer using Snowflake Cortex."""
    prompt = f"""
You are an analyst assistant specialized in interpreting FOMC meeting documents.
You will answer questions using ONLY the provided excerpts.
If the excerpts do not include an answer, say so clearly.

Question:
{query}

Context:
{context}

Answer directly, clearly, and factually:
"""
    try:
        return complete(MODEL_NAME, prompt)
    except Exception as e:
        st.error(f"❌ Generation error: {e}")
        return None


# ----------------------------
# UI
# ----------------------------
st.title("📊 FOMC RAG Chatbot (Snowflake Cortex)")
st.markdown("Ask questions about FOMC documents, projections, or inflation data.")

query = st.text_input("Enter your question:", placeholder="e.g. What were the inflation expectations for 2025?")

if query:
    with st.spinner("🔍 Retrieving relevant excerpts..."):
        retrieved = retrieve_context(query)

    if not retrieved:
        st.warning("No relevant context found. Try rephrasing your question.")
    else:
        # Prepare combined context
        context_text = "\n\n".join([r["chunk"] for r in retrieved])
        with st.spinner("🤖 Generating answer..."):
            answer = generate_answer(query, context_text)

        if answer:
            st.subheader("🤖 Answer")
            st.write(answer)

        # Expandable section for context
        with st.expander("🔍 View Retrieved Context"):
            for i, r in enumerate(retrieved, 1):
                source = r.get("source", "Unknown Source")
                st.markdown(f"**Excerpt {i}** — _{source}_")
                st.write(r["chunk"])
                st.divider()

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Built with Snowflake Cortex Search + Streamlit • Improved by GPT-5")
