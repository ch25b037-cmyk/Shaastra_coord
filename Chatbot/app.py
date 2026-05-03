import streamlit as st
from tree_builder import build_tree
from retrieval import HybridRetriever
from traversal import dfs
from groq import Groq

st.set_page_config(page_title="Infinity War Chatbot", page_icon="🪨", layout="centered")

def load_css():
    with open("style.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
load_css()


@st.cache_resource
def load_retriever():
    root, nodes = build_tree("infinity_war.json")
    retriever = HybridRetriever(nodes)
    return root, nodes, retriever


def build_context(nodes):
    return "\n\n".join([f"{n.title}: {n.text}" for n in nodes])


GREETINGS = {'hello', 'hi', 'hey', 'greetings', "what's up", "how are you"}

st.title("Infinity War Chatbot")
st.caption("Ask anything about Avengers: Infinity War")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🪨" if msg["role"] == "assistant" else "🦸"):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask about Thanos, the Snap, Vormir, Titan...")

if query:
    # Show user message
    with st.chat_message("user", avatar="🦸"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Greeting shortcut
    if query.lower().strip() in GREETINGS:
        reply = "Hello! Ask me anything about Avengers: Infinity War ⚡"
        with st.chat_message("assistant", avatar="🪨"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.stop()

    # Load retriever (cached)
    with st.status("Assembling the Infinity Stones...", expanded=True) as status:
        st.write("Loading knowledge tree...")
        root, nodes, retriever = load_retriever()

        st.write("Retrieving relevant nodes...")
        results, top_score = retriever.retrieve(query)

        if top_score < 0.5:
            status.update(label="Done", state="complete", expanded=False)
            reply = "I'm not confident I have info on that. Try asking about Thanos, Wakanda, Titan, or Vormir!"
            with st.chat_message("assistant", avatar="🪨"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.stop()

        st.write("Expanding context with DFS traversal...")
        expanded = []
        seen = set()
        for n in results:
            for node in dfs(n, depth=1):
                if node.node_id not in seen:
                    expanded.append(node)
                    seen.add(node.node_id)

        context = build_context(expanded)
        status.update(label="Context ready — generating answer...", state="running")

        prompt = f"""You are an AI assistant for Avengers Infinity War.
Use ONLY the context below to answer. Be concise and clear.

Context:
{context}

Question:
{query}

Answer:"""

        client = Groq(api_key=st.secrets["api_key"])

        # Stream the response
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            stream=True
        )

        status.update(label="Done", state="complete", expanded=False)

    # Stream output into chat message
    with st.chat_message("assistant", avatar="🪨"):
        response = st.write_stream(
            chunk.choices[0].delta.content or ""
            for chunk in stream
        )

    st.session_state.messages.append({"role": "assistant", "content": response})