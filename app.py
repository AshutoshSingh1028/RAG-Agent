import streamlit as st
from agent import app, GraphState # Import the compiled app from agent.py
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Medical RAG Agent")

# Check if keys are set
if not all([os.getenv("PINECONE_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("GROQ_API_KEY")]):
    st.error("Missing API keys! Please check your .env file.")
    st.stop()

# STREAMLIT UI

st.set_page_config(page_title="Medical RAG Agent", layout="wide")
st.title("🩺 Medical RAG Agent")
st.markdown("""
Ask a question about medical topics. The agent will first decide if it needs to 
search its knowledge base (a medical PDF) or if it can answer directly.
""")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What are the side effects of Aspirin?"):
  
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # RUN THE AGENT
    
    inputs = {
        "question": prompt,
        "answer_attempts": 0
    }

    with st.spinner("The agent is thinking... (Planning, Retrieving, Answering, Reflecting)"):
        try:
            final_state = app.invoke(inputs)
            
            answer = final_state.get("answer", "I'm sorry, I couldn't find an answer.")
            reflection = final_state.get("reflection", "N/A")
            
            # Display the agent's response
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.info(f"**Agent's self-reflection:** {reflection}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"An error occurred while running the agent: {e}")
            print(f"Error invoking agent: {e}") 