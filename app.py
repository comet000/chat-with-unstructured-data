import streamlit as st
from snowflake.snowpark import Session
from snowflake.cortex import Complete
from snowflake.cortex import CortexSearchRetriever

# --- Streamlit page setup ---
st.set_page_config(page_title="FOMC Analyst Chatbot", layout="wide")

st.title("📈 FOMC Analyst Chatbot")
st.caption("Ask questions about Federal Reserve FOMC documents (Jan 2023 – Oct 2025).")

# --- Connection parameters ---
connection_parameters = {
    "account": "fokiamm-yqb60913",
    "user": "streamlit_demo_user",
    "password": "RagCortex#78_Pw",
    "warehouse": "CORTEX_SEARCH_TUTORIAL_WH",
    "database": "CORTEX_SEARCH_TUTORIAL_DB",
    "schema": "PUBLIC",
    "role": "STREAMLIT_READONLY_ROLE",
}

# --- Initialize Snowflake session ---
session = Session.builder.configs(connection_parameters).create()

# --- Create retriever from Cortex Search Service ---
retriever = CortexSearchRetriever(
    snowpark_session=session,
    service_name="FOMC_SEARCH_SERVICE",
    limit_to_retrieve=5  # limit to top 5 most relevant chunks
)

# --- Define expert-style system prompt ---
SYSTEM_PROMPT = """
You are an expert Federal Reserve analyst summarizing FOMC materials.
Respond clearly and confidently to user questions using retrieved documents.
Your answers should sound like a professional economic briefing, not a chatbot.

Follow this structure:
1. Begin with a concise, 2–3 sentence expert summary.
2. Then, provide 2–4 short, relevant supporting excerpts using bullet points.
3. Do not mention 'Source:' or 'Excerpt #' or list long paragraphs.
4. Focus on what the FOMC *expected, decided, or stated*, not speculation.
5. Write in a formal but natural tone.
"""

# --- Chat input ---
query = st.text_input("💬 Ask a question about FOMC policy, inflation, or rates:")

if query:
    with st.spinner("Analyzing FOMC materials..."):
        # Retrieve relevant documents
        docs = retriever.retrieve(query)

        # Combine retrieved text
        context_text = "\n\n".join([doc["chunk"].strip() for doc in docs])

        # Prepare full prompt for Cortex completion
        full_prompt = f"""{SYSTEM_PROMPT}

Question:
{query}

Relevant Context:
{context_text}

Answer:
"""

        # Generate completion using Cortex
        response = Complete(
            model="snowflake-arctic-embed-l-v2.0",
            prompt=full_prompt
        )

        # --- Display results ---
        st.markdown("### 📊 Expert Summary")
        st.write(response["completion"].strip())

        # Option to view the retrieved context (toggle)
        with st.expander("🔍 View Retrieved Context"):
            for doc in docs:
                snippet = doc["chunk"].strip()
                st.markdown(f"- *{snippet[:500]}...*")

else:
    st.info("Type a question above to begin (e.g., *What was the Fed’s rate policy in 2024?*).")

