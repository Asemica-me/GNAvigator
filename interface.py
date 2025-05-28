import asyncio
import streamlit as st
from dotenv import load_dotenv
import os
import time
import pandas as pd
from pathlib import Path
import logging

# Set environment variable to disable problematic watcher
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
logger = logging.getLogger(__name__)

# --- Feedback Handling ---
def save_feedback(feedback_data: dict):
    """Save feedback to CSV file"""
    csv_path = Path("feedback.csv")
    df = pd.DataFrame([feedback_data])
    df.to_csv(
        csv_path, 
        mode="a", 
        header=not csv_path.exists(), 
        index=False
    )

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Chatbot GNA", page_icon="🤖")
    st.title("Geoportale Nazionale Archeologia - Assistente Virtuale")
    
    # Initialize RAG system - Delayed import to avoid PyTorch conflict
    if "orchestrator" not in st.session_state:
        if not MISTRAL_API_KEY:
            st.error("MISTRAL_API_KEY non trovata. Impostare la variabile d'ambiente.")
            st.stop()
        
        # Import after Streamlit initialization to avoid conflict
        from llm_handler import RAGOrchestrator
        st.session_state.orchestrator = RAGOrchestrator(mistral_api_key=MISTRAL_API_KEY)
    
    orchestrator = st.session_state.orchestrator

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = []

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Add feedback for assistant messages
            if role == "assistant":
                # Check if feedback already exists
                feedback_exists = any(fb["message_index"] == i for fb in st.session_state.feedback_data)
                current_rating = next(
                    (fb["rating"] for fb in st.session_state.feedback_data if fb["message_index"] == i), 
                    None
                )
                
                if feedback_exists:
                    st.success(f"✅ Valutazione registrata: {current_rating}/5")
                else:
                    st.caption("Valuta questa risposta:")
                    cols = st.columns(5)
                    for score in range(1, 6):
                        if cols[score-1].button(
                            f"⭐{score}", 
                            key=f"feedback_{i}_{score}",
                            use_container_width=True
                        ):
                            feedback = {
                                "message_index": i,
                                "question": st.session_state.chat_history[i-1]["content"],
                                "answer": content,
                                "rating": score,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.feedback_data.append(feedback)
                            save_feedback(feedback)
                            st.experimental_rerun()

    # User input
    if prompt := st.chat_input("Cosa vuoi chiedere?"):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get RAG response
        with st.spinner("Elaboro la risposta..."):
            # Wrap async call in synchronous context
            try:
                response = asyncio.run(orchestrator.query(question=prompt))
                answer = response["answer"]
            except Exception as e:
                logger.error(f"Query failed: {str(e)}")
                answer = "Si è verificato un errore durante l'elaborazione della richiesta."
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer
        })
        
        # Display assistant response immediately
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # Rerun to show feedback buttons
        st.experimental_rerun()
    
    # Sidebar for additional info
    with st.sidebar:
        st.header("Informazioni")
        st.markdown("""
        **Assistente Virtuale per il Geoportale Nazionale Archeologia**  
        Risponde a domande basate sulla documentazione ufficiale disponibile su:
        [gna.cultura.gov.it](https://gna.cultura.gov.it/wiki/index.php/Pagina_principale)
        """)
        
        if st.session_state.feedback_data:
            st.subheader("Feedback Raccolti")
            st.write(f"Totale feedback: {len(st.session_state.feedback_data)}")
            avg_rating = sum(fb["rating"] for fb in st.session_state.feedback_data) / len(st.session_state.feedback_data)
            st.metric("Valutazione Media", f"{avg_rating:.1f}/5")
            
            if st.button("Esporta tutti i feedback"):
                df = pd.DataFrame(st.session_state.feedback_data)
                st.download_button(
                    label="Scarica CSV",
                    data=df.to_csv(index=False),
                    file_name="feedback_full.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    # Create a new event loop for asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main()