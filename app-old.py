import streamlit as st
import asyncio
from llm_handler import RAGOrchestrator
import pandas as pd
from pathlib import Path
import time
import os
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

async def main():
    if not MISTRAL_API_KEY:
        st.error("MISTRAL_API_KEY not found in environment variables.")
        return

    if "rag_orchestrator" not in st.session_state:
        st.session_state.rag_orchestrator = RAGOrchestrator(mistral_api_key=MISTRAL_API_KEY)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = []

    st.title("Assistente AI per il Geoportale Nazionale Archeologia")

    user_question = st.text_input("Cosa vuoi chiedere?", key="user_question")
    if st.button("Invia", key="send_button"):
        user_question = st.session_state.user_question.strip()
        if not user_question:
            st.warning("Per favore, inserisci una domanda.")
            return

    async def handle_query(question: str):
        if "last_request_time" in st.session_state:
            elapsed = time.time() - st.session_state.last_request_time
            if elapsed < 2.0:
                st.warning("Attendi almeno 2 secondi tra una richiesta e l'altra")
                return

        try:
            with st.spinner("Elaborando la risposta..."):
                response = await st.session_state.rag_orchestrator.query(question=question)
                answer = response.get("answer", "Nessuna risposta trovata.")
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_request_time = time.time()
        except Exception as e:
            st.error(f"Si è verificato un errore durante la query: {e}")
            print(f"Errore nella query: {e}")

    if user_question:
        await handle_query(user_question)

    # Display chat history with feedback
    human_style = "background-color: #3f444f; border-radius: 10px; padding: 10px;"
    chatbot_style = "border-radius: 10px; padding: 10px;"

    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(
                f"<p style='text-align: right;'><b>Utente:</b></p>"
                f"<p style='text-align: right; {human_style}'><i>{message['content']}</i></p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align: left;'><b>Assistente AI:</b></p>"
                f"<p style='text-align: left; {chatbot_style}'><i>{message['content']}</i></p>",
                unsafe_allow_html=True,
            )
            # Feedback section
            feedback_exists = False
            current_rating = None
            for fb in st.session_state.feedback_data:
                if fb["message_index"] == i:
                    feedback_exists = True
                    current_rating = fb["rating"]
                    break

            if feedback_exists:
                st.markdown(
                    f"<p style='text-align: left; color: green;'>Valutazione: {current_rating}/5</p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<p style='text-align: left;'>Valuta questa risposta (1 = Scarso, 5 = Eccellente):</p>",
                    unsafe_allow_html=True
                )
                cols = st.columns(5)
                for score in range(1, 6):
                    if cols[score - 1].button(
                        str(score),
                        key=f"feedback_{i}_{score}"
                    ):
                        # Capture conversation context
                        question = st.session_state.chat_history[i - 1]["content"]
                        answer = message["content"]

                        # Create feedback entry
                        new_feedback = {
                            "message_index": i,
                            "question": question,
                            "answer": answer,
                            "rating": score,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # Update feedback data
                        st.session_state.feedback_data = [
                            fb for fb in st.session_state.feedback_data
                            if fb["message_index"] != i
                        ]
                        st.session_state.feedback_data.append(new_feedback)

                        # Save to CSV
                        feedback_folder = Path("feedback")
                        feedback_folder.mkdir(parents=True, exist_ok=True)
                        csv_path = feedback_folder / "feedback.csv"
                        df = pd.DataFrame([new_feedback])
                        df.to_csv(
                            csv_path,
                            mode="a",
                            header=not csv_path.exists(),
                            index=False
                        )
                        df = pd.DataFrame([new_feedback])
                        df.to_csv(
                            csv_path,
                            mode="a",
                            header=not csv_path.exists(),
                            index=False
                        )
                        st.success("Grazie per il tuo feedback!")
                        st.rerun()

if __name__ == "__main__":
    asyncio.run(main())