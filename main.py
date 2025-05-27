import os
from dotenv import load_dotenv
import streamlit as st
import mistralai
from crawl_kb import *
from create_chunks import *
from create_embeddings_json import *
from create_vectorstore import *
from process_query import *

async def main():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("MODEL")
    embedding_model_name = "mistral-embed"

    if "mistral_client" not in st.session_state:
        st.session_state.mistral_client = Mistral(api_key=api_key)

    client = st.session_state.mistral_client
    st.session_state.client_data = (client, model_name)

    system_prompt = """
        Sei un assistente virtuale incaricato di rispondere a domande sul manuale operativo del Geoportale Nazionale Archeologia (GNA), disponibile all'indirizzo: https://gna.cultura.gov.it/wiki/index.php/Pagina_principale, e gestito dall'Istituto centrale per il catalogo e la documentazione (ICCD).

        Rispondi **sempre** in italiano, indipendentemente dalla lingua della domanda, a meno che l'utente non richieda esplicitamente un'altra lingua.

        Devi sempre rispondere in modo disponibile, professionale e naturale. Le tue risposte devono essere grammaticalmente corrette e coerenti, evitando frasi complesse o frammentate.

        Non rispondere a una domanda con un'altra domanda. Rispondi sempre in modo completo e chiaro, evitando di lasciare domande senza risposta.

        Quando rispondi, fai riferimento al seguente contesto: 
        {context}
        """
    st.session_state.system_prompt_template = system_prompt
    
    st.set_page_config(
        page_title="Assistente virtuale GNA STREAM",
        page_icon=":bust_in_silhouette:"
    )

    # Initialize components
    if "client_data" not in st.session_state:
        st.session_state.client_data = (client, model_name)
    
    if "vector_store" not in st.session_state:
        metadata_df = await crawl_metadata_async('GNA_sitemap.xml')
        st.session_state.vector_store = FAISS.from_texts(
            metadata_df['text'].tolist(),
            embedding=MistralAIEmbeddings(api_key=api_key),
            metadatas=metadata_df[['url', 'title']].to_dict('records')
        )

    if "conversation" not in st.session_state:
        memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

    if "chat_history" in st.session_state:
        converted_history = []
        for msg in st.session_state.chat_history:
            if isinstance(msg, dict):
                if msg["type"] == "user":
                    converted_history.append(HumanMessage(content=msg["content"]))
                else:
                    converted_history.append(AIMessage(content=msg["content"]))
            else:
                converted_history.append(msg)
        st.session_state.chat_history = converted_history
    else:
        st.session_state.chat_history = []

    # Interfaccia utente
    st.title("Assistente Virtuale GNA")

    st.markdown(
        """
        Questo assistente è stato creato per rispondere a domande sul manuale d'uso per il progetto [Geoportale Nazionale dell’Archeologia (GNA)](https://gna.cultura.gov.it/wiki/index.php/Pagina_principale).
        Poni una domanda e l'assistente ti risponderà con il contenuto più rilevante del manuale.
        """
    )

    # Immagine
    st.markdown(
        """
        <style>
        .rounded-image {
            border-radius: 15px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<img src="https://github.com/Asemica-me/chatw-GNA/blob/main/data/img.jpg?raw=true" class="rounded-image" alt="Chatbot Assistance"/>', 
        unsafe_allow_html=True
    )


    user_input = st.text_input("Cosa vuoi chiedere?")
    if user_input:
        with st.spinner("Elaborando la risposta..."):
                try:
                    response = await process_query(
                        st.session_state.vector_store,
                        st.session_state.client_data,
                        system_prompt,
                        user_input,
                        st.session_state.chat_history
                    )
                    
                    st.session_state.chat_history = [
                        msg for msg in st.session_state.chat_history 
                        if not (isinstance(msg, dict) or (hasattr(msg, 'content') and msg.content.strip() == ""))
                    ]

                    handle_style_and_responses(user_input)
                    
                except Exception as e:
                    st.error(f"Error while elaborating: {str(e)}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())