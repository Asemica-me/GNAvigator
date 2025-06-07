import streamlit as st
from PIL import Image
import os
import time
import logging
import gc
import pandas as pd
import re
from feedback_handling import init_db, git_sync, save_feedback, export_feedbacks, FEEDBACK_DB
init_db()

# --- Critical Dependencies Setup ---
# Create directories first (runs once per session)
USER_DATA_DIR = "/tmp/nlp_data"
os.makedirs(f"{USER_DATA_DIR}/spacy", exist_ok=True)
os.makedirs(f"{USER_DATA_DIR}/nltk", exist_ok=True)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SPACY_DATA_DIR", f"{USER_DATA_DIR}/spacy")
os.environ.setdefault("NLTK_DATA", f"{USER_DATA_DIR}/nltk")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Configure page
try:
    icon = Image.open("data/gna.png")
    st.set_page_config(
        page_title="Assistente AI GNA",
        page_icon=icon
    )
except Exception:
    st.set_page_config(
        page_title="Assistente AI GNA",
        page_icon="🤖"
    )
st.title("Geoportale Nazionale Archeologia - Assistente Virtuale")

# --- Cached Resource Loaders ---
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """Load spaCy model with caching and download if missing"""
    import spacy
    try:
        return spacy.load("it_core_news_md")
    except OSError:
        st.warning("Download modello spaCy italiano...")
        import subprocess
        import sys
        
        # Try direct download
        try:
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "it_core_news_md"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return spacy.load("it_core_news_md")
            raise RuntimeError(result.stderr)
        except Exception:
            # Fallback to direct URL
            try:
                model_url = "https://github.com/explosion/spacy-models/releases/download/it_core_news_md-3.8.0/it_core_news_md-3.8.0-py3-none-any.whl"
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir", model_url],
                    check=True
                )
                return spacy.load("it_core_news_md")
            except Exception as e:
                st.error(f"Installazione spaCy fallita: {str(e)}")
                st.stop()

@st.cache_resource(show_spinner=False)
def load_nltk_data():
    """Download required NLTK data"""
    import nltk
    try:
        nltk.download("punkt", download_dir=os.environ["NLTK_DATA"])
        nltk.data.path.append(os.environ["NLTK_DATA"])
    except Exception as e:
        st.error(f"Configurazione NLTK fallita: {str(e)}")

@st.cache_resource(show_spinner=False)
def load_opencv():
    """Install OpenCV if missing"""
    try:
        import cv2
        return cv2
    except ImportError:
        st.warning("Installing OpenCV...")
        import subprocess
        import sys
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "opencv-python-headless"],
                check=True
            )
            import cv2
            return cv2
        except Exception as e:
            st.error(f"OpenCV install failed: {str(e)}")
            st.stop()

# Initialize NLP resources
try:
    nlp = load_spacy_model()
    load_nltk_data()
    cv2 = load_opencv()
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

# --- Main Application ---
from dotenv import load_dotenv
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def format_answer_with_links(answer: str, source_map: dict) -> str:
    """Convert [number] citations to markdown hyperlinks"""
    def replace_citation(match):
        citation_num = match.group(1)
        if citation_num in source_map:
            url = source_map[citation_num]
            return f"[[{citation_num}]]({url})"
        return match.group(0)
    return re.sub(r'\[\'?(\d+)\'?\]', replace_citation, answer)

# --- Main Function with memory optimizations ---
def main():
    import logging
    MAX_HISTORY = 10  # Limit chat history entries
    GC_INTERVAL = 3  # Garbage collection interval in interactions
    
    # Initialize feedback database
    init_db()
    
    # Initialize orchestrator with caching
    @st.cache_resource(show_spinner="Caricamento del modello...")
    def get_orchestrator():
        if not MISTRAL_API_KEY:
            st.error("MISTRAL_API_KEY mancante!")
            st.stop()
        from llm_handler import RAGOrchestrator
        return RAGOrchestrator(mistral_api_key=MISTRAL_API_KEY)
    
    
    orchestrator = get_orchestrator()

    if "last_cleanup" not in st.session_state:
        st.session_state.last_cleanup = time.time()

    # Clear every 10 minutes
    if time.time() - st.session_state.last_cleanup > 600:
        orchestrator.clear_cache()
        st.session_state.last_cleanup = time.time()

    # Initialize session state with size limits
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.interaction_count = 0
    
    # Feedback tracking - store only minimal data
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

    # Display chat history with truncation
    for i, message in enumerate(st.session_state.chat_history[-MAX_HISTORY:]):
        idx_offset = max(0, len(st.session_state.chat_history) - MAX_HISTORY)
        adjusted_idx = i + idx_offset
        
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Feedback for assistant messages
            if message["role"] == "assistant":
                if adjusted_idx in st.session_state.feedback_given:
                    st.success("✅ Valutazione registrata")
                else:
                    st.caption("Valuta questa risposta:")
                    cols = st.columns(3)
                    for score in range(1, 4):
                        if cols[score-1].button(
                            f"⭐{score}", 
                            key=f"feedback_{adjusted_idx}_{score}",
                            use_container_width=True
                        ):
                            feedback = {
                                "message_index": adjusted_idx,
                                "question": st.session_state.chat_history[adjusted_idx-1]["content"],
                                "answer": message["raw_answer"],
                                "rating": score,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            save_feedback(feedback)
                            st.session_state.feedback_given.add(adjusted_idx)
                            st.rerun()

    # User input
    if prompt := st.chat_input("Cosa vuoi chiedere?"):

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.interaction_count += 1
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.spinner("Elaboro la risposta..."):
            import asyncio
            try:
                # Prepare chat history for LLM (only role and content)
                llm_history = [
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in st.session_state.chat_history
                    if "raw_answer" not in msg  # Exclude assistant's raw answer
                ]
                
                response = asyncio.run(orchestrator.query(
                    question=prompt, 
                    chat_history=llm_history,  # Pass conversation history
                    top_k=5
                ))
                source_map = response.get("sources", {})
                raw_answer = response["answer"]
                formatted_answer = format_answer_with_links(raw_answer, source_map)
            except Exception as e:
                logging.error(f"Query failed: {str(e)}")
                error_msg = f"Si è verificato un errore durante l'elaborazione: {type(e).__name__}\n\n{str(e)}"
                raw_answer = error_msg
                formatted_answer = error_msg 
        
        # Add assistant response
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": formatted_answer,
            "raw_answer": raw_answer
        })
        
        # Apply history limit
        if len(st.session_state.chat_history) > MAX_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]

        # Force garbage collection periodically
        if st.session_state.interaction_count % GC_INTERVAL == 0:
            gc.collect()
        
        # update UI
        st.rerun()
    
    # Sidebar
    st.sidebar.image("data/gna.png", width=60)
    with st.sidebar:
        st.header("Informazioni")
        st.markdown(f"""
        **Assistente AI per il Geoportale Nazionale Archeologia**
        
        Questo assistente virtuale utilizza tecnologie di Intelligenza Artificiale per rispondere a domande relative al Geoportale Nazionale Archeologia (GNA).
        
        Il modello è stato addestrato sul manuale operativo e sulla documentazione ufficiale del progetto, gestito dall'Istituto Centrale per il Catalogo e la Documentazione (ICCD) e disponibile a questo indirizzo:
            """)
        st.markdown(f"""
        [gna.cultura.gov.it](https://gna.cultura.gov.it/wiki/index.php/Pagina_principale)""", unsafe_allow_html=True)
            
        st.divider()

        # Add clear chat history button
        if st.button("Cancella cronologia chat"):
            st.session_state.chat_history = []
            st.rerun()
        st.caption("Gestione feedback")
        st.markdown("""
        Lascia un feedback sulle risposte dell'assistente per migliorare le sue prestazioni.
        """)

        st.divider()
            
        # Feedback export
        if st.button("Esporta feedback"):
            try:
                df = export_feedbacks()
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Scarica CSV",
                        data=csv,
                        file_name="feedbacks_assistenteAI_gna.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Nessun feedback disponibile")
            except Exception as e:
                st.error(f"Errore durante l'esportazione: {str(e)}")

if __name__ == "__main__":
    main()