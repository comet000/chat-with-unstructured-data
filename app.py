from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.cortex import complete
import logging

# --- Initialize session state variables safely ---
# Initialize session state keys
if "messages" not in st.session_state:
st.session_state.messages = []

if "rag_cache" not in st.session_state:
st.session_state.rag_cache = {}

st.set_page_config(page_title="Chat with the Federal Reserve", page_icon="💬", layout="centered")
st.title("🏦 Chat with the Federal Reserve")
st.markdown("**Built on 5000 pages of Fed documents from 2023 - 2025**")

# --- Snowflake session creation ---
# Hide Streamlit default menu and footer for cleaner UI
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Snowflake connection
@st.cache_resource
def create_snowflake_session():
connection_parameters = {
@@ -110,6 +119,7 @@ def retrieve(self, query: str) -> List[dict]:
return [{"chunk": d["chunk"], "file_name": d["file_name"]} for d in docs]

except Exception as e:
            logging.error(f"Cortex Search retrieval error: {e}")
st.error(f"❌ Cortex Search Error: {e}")
return []

@@ -183,26 +193,16 @@ def generate_response_stream(query: str, contexts: List[dict]):
st.session_state.rag_cache.clear()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if msg["role"] != "system":  # Hide system messages from user
        st.chat_message(msg["role"]).write(msg["content"])

def run_query(user_query: str):
with st.spinner("Searching..."):
contexts = rag_retriever.retrieve(user_query)

    with st.expander("🔍 See Retrieved Context"):
        if not contexts:
            st.info("No relevant context found.")
        for item in contexts:
            title = extract_clean_title(item["file_name"])
            pdf_url = create_direct_link(item["file_name"])
            snippet = clean_chunk(item["chunk"])[:600]
            snippet += "..." if len(item["chunk"]) > 600 else ""
            st.markdown(f"**{title}**")
            st.write(snippet)
            st.markdown(f"[📄 View Full Document]({pdf_url})")

if not contexts:
        return ["No relevant context found."]
        st.info("No relevant context found.")
        return ["No relevant documents found."]

st.session_state.messages.append({"role": "system", "content": build_system_prompt(user_query, contexts)})
with st.spinner("Generating response..."):
