import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from tavily import TavilyClient
import streamlit as st

load_dotenv()

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Pro Search AI", page_icon="🌐", layout="wide")

# 2. CUSTOM CSS FOR A MODERN LOOK
st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .stChatMessage { border: 1px solid #30363d; border-radius: 10px; padding: 10px; margin-bottom: 15px; }
        .stMarkdown p { font-size: 1.1rem; color: #e6edf3; }
        div[data-testid="stToolbar"] { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# 3. SIDEBAR CONTROLS
with st.sidebar:
    st.title("AI Settings")
    st.divider()
    model_name = st.selectbox("Choose Model", ["llama-3.3-70b-versatile"])
    temp = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.info("Powered by **Groq** and **Tavily**")

# 4. INITIALIZE LLM & SEARCH TOOL
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool("brave_search", description="Search the web for real-time information.")
def internet_search(user_query: str):
    return tavily_client.search(user_query)


llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=temp)
model = llm.bind_tools([internet_search])

# 5. INITIALIZE CHAT MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. DISPLAY CHAT INTERFACE
st.title("Pro Search Assistant")
st.caption("Ask me anything — I'll browse the web for the latest updates.")

# Show previous messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="👤").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant", avatar="🤖").write(msg.content)

# 7. CHAT LOGIC
if prompt := st.chat_input("What would you like to know?"):
    # Show user message
    st.chat_message("user", avatar="👤").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar="🤖"):
        # We use a container to show progress clearly
        status_placeholder = st.empty()

        with st.spinner("Analyzing and Searching..."):
            # Step 1: Send history to AI
            response = model.invoke(st.session_state.messages)
            st.session_state.messages.append(response)

            # Step 2: If the AI wants to search
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    status_placeholder.status(f"🔍 Searching for: {tool_call['args']['user_query']}")

                    search_results = internet_search.invoke(tool_call["args"])
                    tool_msg = ToolMessage(content=str(search_results), tool_call_id=tool_call["id"])
                    st.session_state.messages.append(tool_msg)

                # Step 3: Get the final answer with search context
                final_response = model.invoke(st.session_state.messages)
                st.write(final_response.content)
                st.session_state.messages.append(final_response)
            else:
                # Direct answer if no search was needed
                st.write(response.content)

# 8. AUTO-SCROLL FIX
st.markdown('<div id="end"></div>', unsafe_allow_html=True)
