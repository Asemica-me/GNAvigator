# feedback collection and storage

import time
import streamlit as st
import pandas as pd
from pathlib import Path

def handle_style_and_responses(user_question: str, mistral_llm) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit.
    Includes feedback collection and storage.
    """
    if "last_request_time" in st.session_state:
        elapsed = time.time() - st.session_state.last_request_time
        if elapsed < 2.0:
            st.warning("Attendi almeno 2 secondi tra una richiesta e l'altra")
            return

    try:
        # Initialize session state variables
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        if "feedback_data" not in st.session_state:
            st.session_state.feedback_data = []

        # Get model response
        response = st.session_state.conversation.invoke({"question": user_question})
        st.session_state.chat_history = response.get("chat_history", [])
        st.session_state.last_request_time = time.time()

        # Define styles
        human_style = "background-color: #3f444f; border-radius: 10px; padding: 10px;"
        chatbot_style = "border-radius: 10px; padding: 10px;"

        # Display messages with feedback
        for i, message in enumerate(st.session_state.chat_history):
            if not hasattr(message, "content") or not message.content:
                st.warning(f"Messaggio non valido alla posizione {i}: {message}")
                continue

            if i % 2 == 0:  # User message
                st.markdown(
                    f"<p style='text-align: right;'><b>Utente:</b></p>"
                    f"<p style='text-align: right; {human_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )
            else:  # Assistant message
                st.markdown(
                    f"<p style='text-align: left;'><b>Assistente AI:</b></p>"
                    f"<p style='text-align: left; {chatbot_style}'><i>{message.content}</i></p>",
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
                        "<p style='text-align: left;'>Valuta questa risposta (1=Scarso, 5=Eccellente):</p>", 
                        unsafe_allow_html=True
                    )
                    cols = st.columns(5)
                    for score in range(1, 6):
                        if cols[score-1].button(
                            str(score),
                            key=f"feedback_{i}_{score}"
                        ):
                            # Capture conversation context
                            question = st.session_state.chat_history[i-1].content
                            answer = message.content
                            
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
                            csv_path = Path("feedback.csv")
                            df = pd.DataFrame([new_feedback])
                            df.to_csv(
                                csv_path, 
                                mode="a", 
                                header=not csv_path.exists(), 
                                index=False
                            )
                            
                            st.success("Grazie per il tuo feedback!")
                            st.rerun()

    except Exception as e:
        st.error(f"Si è verificato un errore: {str(e)}")
        print(f"Errore nella gestione della risposta: {e}")

# Example usage in main Streamlit app:
# st.title("Chatbot")
# user_question = st.text_input("Fai una domanda...")
# if user_question:
#     handle_style_and_responses(user_question, mistral_llm)

# Pandas analusis of feedback data
# df = pd.read_csv("feedback.csv")
# print(df.describe())  # Basic statistics
# print(df.groupby('rating').count())  # Rating distribution