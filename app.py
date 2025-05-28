import asyncio
import streamlit as st
from dotenv import load_dotenv
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import time
import pandas as pd
from pathlib import Path
import logging

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
logger = logging.getLogger(__name__)

# --- Feedback Handling ---
def save_feedback(feedback_data: dict):
    """Save feedback to CSV file in feedback directory"""
    try:
        # Create feedback directory with absolute path
        feedback_dir = Path(__file__).parent / "feedback"
        logger.info(f"Creating feedback directory at: {feedback_dir.absolute()}")
        
        # Create directory (with parents) if it doesn't exist
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Set CSV path
        csv_path = feedback_dir / "feedbacks.csv"
        logger.info(f"Saving feedback to: {csv_path}")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame([feedback_data])
        header = not csv_path.exists()
        
        df.to_csv(
            csv_path, 
            mode="a", 
            header=header, 
            index=False,
            encoding="utf-8"
        )
        logger.info(f"Feedback saved successfully: {feedback_data}")
        
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}", exc_info=True)
        # Attempt to log error to file in current directory as fallback
        try:
            error_path = Path("feedback_errors.log")
            with error_path.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: {e}\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - DATA: {feedback_data}\n")
        except Exception as fallback_error:
            logger.critical(f"Fallback logging failed: {fallback_error}")

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Assistente AI GNA", page_icon="🤖")
    st.title("Geoportale Nazionale Archeologia - Assistente Virtuale")
    
    # Initialize RAG system - Delayed import to avoid PyTorch conflict
    if "orchestrator" not in st.session_state:
        if not MISTRAL_API_KEY:
            st.error("MISTRAL_API_KEY non trovata. Impostare la variabile d'ambiente.")
            st.stop()
        
        # Import ONLY when needed
        with st.spinner("Caricamento del modello..."):
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
                            st.rerun()

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
        st.rerun()
    
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
                try:
                    feedback_dir = Path(__file__).parent / "feedback"
                    csv_path = feedback_dir / "feedbacks.csv"
                    
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        st.download_button(
                            label="Scarica CSV",
                            data=df.to_csv(index=False),
                            file_name="feedback_full.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Nessun feedback disponibile per l'esportazione")
                except Exception as e:
                    st.error(f"Errore durante l'esportazione: {str(e)}")

if __name__ == "__main__":
    main()