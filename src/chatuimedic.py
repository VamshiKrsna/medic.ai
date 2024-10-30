import streamlit as st
from allatonce import MedicAI
import os
from dotenv import load_dotenv

if 'medic_ai' not in st.session_state:
    st.session_state.medic_ai = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_chain():
    """Initialize RAG chain and store in session state"""
    if st.session_state.rag_chain is None:
        with st.spinner("Initializing AI... Please wait."):
            load_dotenv()
            medic_ai = MedicAI('..//Data/')
            api_key = os.getenv("PINECONE_API_KEY")
            pc = medic_ai.setup_pinecone(api_key=api_key)
            vectorstore = medic_ai.get_docsearch_from_pinecone(
                index_name="medicoai",
                embeddings=medic_ai.embeddings
            )
            llm = medic_ai.setup_local_gemma()
            st.session_state.rag_chain = medic_ai.setup_rag_chain(
                llm=llm[1],
                vectorstore=vectorstore
            )

def display_chat():
    """Display chat messages from session state chat history"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

def get_user_input():
    """Get user input and add to chat history immediately upon enter"""
    user_input = st.chat_input("Ask your medical question here...")
    if user_input:
        # Append user query to chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    return user_input

def main():
    st.set_page_config(page_title="Medic AI 🤖🩺", layout="wide")
    st.title("Medic AI 🤖🩺")
    
    # Initialize RAG chain if not already done
    initialize_rag_chain()

    # Display chat history
    display_chat()

    # Capture user input and respond
    user_input = get_user_input()
    if user_input:
        # Stream the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in st.session_state.rag_chain.stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                # Add assistant's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()