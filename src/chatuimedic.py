import streamlit as st
from allatonce import MedicAI
import os
from dotenv import load_dotenv

# session state variables
if 'medic_ai' not in st.session_state:
    st.session_state.medic_ai = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_chain():
    """Initialize RAG chain once and store in session state"""
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

def get_user_input():
    """Get user input from text box"""
    input_container = st.container()
    with input_container:
        user_input = st.chat_input("Ask your medical question here...", key="user_input")
    return user_input

def display_chat_history():
    """Displays chat history"""
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, str):
                if message.startswith("User: "):
                    with st.chat_message("user"):
                        st.write(message[6:])
                elif message.startswith("Bot: "):
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message[5:])
            else:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message["content"])

def main():
    st.set_page_config(
        page_title="Medic AI 🤖🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS 
    st.markdown("""
        <style>
        .main {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .stApp header {
            background-color: white;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
            height: 60px;
            border-bottom: 1px solid #e5e5e5;
        }
        
        .stApp h1 {
            color:black;
            font-size: 24px !important;
            padding: 1rem !important;
            margin: 0 !important;
            background-color: white;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            border-bottom: 1px solid #e5e5e5;
        }
        
        .chat-container {
            margin-top: 80px !important;
            margin-bottom: 100px !important;
            padding: 1rem !important;
            overflow-y: auto;
            center: auto;
        }
        
        .stChatMessage {
            color:black;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            background-color: #f7f7f8;
        }
        
        .stChatMessage[data-testid*="user"] {
            background-color: #f0f7ff;
        }
        
        /* Input area styling */
        .stChatInput {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem;
            background-color: white;
            border-top: 1px solid #e5e5e5;
            z-index: 1000;
        }
        
        /* Input field styling */
        .stChatInput > div {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .stChatInput input {
            border: 1px solid #e5e5e5;
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-size: 1rem;
            background-color: white;
            color: #000000;
        }
        
        .stChatInput input:focus {
            border-color: #0066cc;
            box-shadow: 0 0 0 2px rgba(0,102,204,0.2);
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .stChatMessage {
                max-width: 100%;
                margin: 0.5rem;
            }
            
            .stApp h1 {
                font-size: 20px !important;
            }
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    initialize_rag_chain()
    st.title("Medic AI 🤖🩺")

    main_container = st.container()
    with main_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_chat_history()
        st.markdown('</div>', unsafe_allow_html=True)

    user_input = get_user_input()

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in st.session_state.rag_chain.stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
