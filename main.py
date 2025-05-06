import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import uuid

from mistralai import Mistral
from langchain_mistralai import ChatMistralAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
import streamlit as st
import streamlit.components.v1 as components

import time
import requests
import hashlib

from threading import Lock
import httpx

def load_dataset(dataset_name: str = "gna_kg_dataset_new.csv") -> pd.DataFrame:
    """Carica un dataset da file CSV."""
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset caricato con successo. Numero di righe: {len(df)}")
        #print(f"Prime righe del dataset:\n{df.head()}")  # Aggiungi per vedere le prime righe
        return df
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        raise

def generate_chunk_id(url: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{url}-{chunk_index}".encode()).hexdigest()

def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int):
    """Crea chunk informativi dal dataset per l'archiviazione e il recupero."""
    print("Creazione dei chunk in corso...")
    text_chunks = DataFrameLoader(
        dataset, page_content_column="body"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
    )

    #print(f"Numero di chunk generati: {len(text_chunks)}")

    formatted_chunks = []
    for i, doc in enumerate(text_chunks):
        title = doc.metadata.get("title", "No Title")
        description = doc.metadata.get("description", "No Description")
        content = doc.page_content[:100] + "..."  # Troncamento per evitare output troppo lungo
        url = doc.metadata.get("url", "No URL")

        # Genera chunk_id
        chunk_id = generate_chunk_id(url, i)
        doc.metadata["chunk_id"] = chunk_id

        final_content = f"TITLE: {title}\nDESCRIPTION: {description}\nBODY: {doc.page_content}\nURL: {url}"
        doc.page_content = final_content
        formatted_chunks.append(doc)

    return formatted_chunks

# During retrieval, include image metadata chunks:
def format_image_answer(image_chunk):
    return f"**Image**: {image_chunk.metadata['caption']}\nURL: {image_chunk.metadata['url']}"


def create_or_get_vector_store(chunks: list, api_key: str) -> FAISS:
    """Crea o carica un vector store FAISS."""
    load_dotenv()

    mistral_embeddings = MistralAIEmbeddings(api_key=api_key, wait_time=3)

    if not os.path.exists("./db"):
        os.makedirs("./db")

    vector_store_path = "./db/index_gna.faiss"
    if not os.path.exists(vector_store_path):
        print("Creazione nuovo vector store...")

        # Add metadata with unique chunk IDs
        processed_chunks = [
            Document(
                page_content=doc.page_content,
                metadata={
                    "chunk_id": str(uuid.uuid4()),  # Generate unique ID
                    **doc.metadata  # Preserve existing metadata
                }
            ) for doc in chunks if doc.page_content.strip()
        ]

        if not processed_chunks:
            raise ValueError("No valid chunks found")

        # Usa from_documents() con gli oggetti Document completi
        vector_store = FAISS.from_documents(processed_chunks, mistral_embeddings)
        vector_store.save_local("./db")
    else:
        vector_store = FAISS.load_local("./db", embeddings=mistral_embeddings, allow_dangerous_deserialization=True)

    return vector_store

    
# Classe per la gestione del rate limiting
class TokenBucketRateLimiter:
    def __init__(self, rate, capacity):
        self.rate = rate  # token al secondo
        self.capacity = capacity  # capacità massima del bucket
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens=1):
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
            time.sleep(0.01)

def create_mistral_client(
    api_key: str, 
    model_name: str):
    """Create a Mistral LLM client with rate limiting"""
    if not model_name:
        raise ValueError(
            "Mistral model name must be provided either via "
            "MISTRAL_MODEL in .env file or model_name parameter"
        )
    print(f"Inizializzazione di Mistral: {model_name}")

    rate_limiter = TokenBucketRateLimiter(
        rate=0.0083,  # 1 request every 120 seconds
        capacity=10
    )

    client = Mistral(api_key=api_key)
    print("Mistral configurato correttamente")
    return client, rate_limiter, model_name

def invoke_with_retry(client, rate_limiter, model_name, messages, 
                     max_retries=5, initial_delay=15.0, backoff_factor=2.5):
    """Improved retry logic with better rate limit handling"""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            rate_limiter.consume()
            response = client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = e.response.headers.get('Retry-After', delay)
                wait_time = float(retry_after) if retry_after else delay
                print(f"Rate limited. Retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                delay *= backoff_factor
            else:
                raise
        except Exception as e:
            print(f"API error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= backoff_factor
    raise Exception("Max retries exceeded")

def process_query(vector_store, client_data, system_prompt, user_question, chat_history):
    """Elabora una query utilizzando il client Mistral"""
    client, rate_limiter, model_name = client_data

    # Rate limit before making the API call
    rate_limiter.consume()
    
    # Ricerca documenti rilevanti
    docs = vector_store.as_retriever().get_relevant_documents(user_question)
    context = "\n".join([doc.page_content for doc in docs][:5])  # Limita a 5 documenti

    # Costruzione messaggi
    messages = [{"role": "system", "content": system_prompt.format(context=context)}]
    
    # Aggiunta cronologia conversazione (modificato)
    for msg in chat_history[-4:]:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict):  # Gestione residuale
            messages.append({"role": msg.get("type", "user"), "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": user_question})

    # Generazione risposta
    response = invoke_with_retry(client, rate_limiter, model_name, messages)
    
    # Aggiornamento cronologia (corretto)
    chat_history.extend([
        HumanMessage(content=user_question), 
        AIMessage(content=response)          
    ])
    
    return response

def handle_style_and_responses(user_question: str) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit.
    """
    try:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        response = st.session_state.conversation.invoke({"question": user_question})
        st.session_state.chat_history = response.get("chat_history", [])

        human_style = "background-color: #3f444f; border-radius: 10px; padding: 10px;"
        chatbot_style = "border-radius: 10px; padding: 10px;"

        for i, message in enumerate(st.session_state.chat_history):
            # Controllo struttura del messaggio
            if not hasattr(message, "content") or not message.content:
                st.warning(f"Messaggio non valido alla posizione {i}: {message}")
                continue

            if i % 2 == 0:  # Messaggi dell'utente
                st.markdown(
                    f"<p style='text-align: right;'><b>Utente:</b></p>"
                    f"<p style='text-align: right; {human_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )
            else:  # Messaggi del chatbot
                st.markdown(
                    f"<p style='text-align: left;'><b>Assistente AI:</b></p>"
                    f"<p style='text-align: left; {chatbot_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        # Logging degli errori
        st.error(f"Si è verificato un errore: {str(e)}")
        print(f"Errore nella gestione della risposta: {e}") 



def main():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("MODEL")

    system_prompt = """
        Sei un assistente virtuale incaricato di rispondere alle domande sul manuale utente WikiMedia del [Geoportale Nazionale dell’Archeologia (GNA)](https://gna.cultura.gov.it/wiki/index.php/Pagina_principale), gestito dall'Istituto centrale per il catalogo e la documentazione (ICCD).

        Rispondi **sempre** in italiano, indipendentemente dalla lingua della domanda, a meno che l'utente non richieda esplicitamente un'altra lingua.

        Devi sempre rispondere in modo amichevole, professionale e utile.
        Le tue risposte devono essere grammaticalmente corrette, coerenti e facili da comprendere, evitando frasi complesse o frammentate.

        Non rispondere a una domanda con un'altra domanda.

        Evita l'uso eccessivo di elenchi puntati, liste numerate, tabelle o formattazioni eccessive nelle tue risposte.
        Le tue risposte devono essere naturali, come se fossero scritte da un essere umano esperto, utilizzando paragrafi ben formati invece di output meccanici o eccessivamente strutturati.

        Assicurati sempre che le tue risposte si basino sui contenuti più rilevanti del manuale utente GNA.

        Quando rispondi, fai riferimento esclusivamente al seguente contesto: 
        {context}
        """
    
    st.set_page_config(
        page_title="Assistente virtuale GNA",
        page_icon=":bust_in_silhouette:"
    )

    # Inizializzazione componenti

    if "client_data" not in st.session_state:
        st.session_state.client_data = create_mistral_client(api_key, model_name)
    
    if "vector_store" not in st.session_state:
        dataset = load_dataset("gna_kg_dataset.csv")
        chunks = create_chunks(dataset, chunk_size=1000, chunk_overlap=200)  # Pass required parameters
        st.session_state.vector_store = create_or_get_vector_store(chunks, api_key)

    if "conversation" not in st.session_state:
    # Initialize the conversation chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=ChatMistralAI(api_key=api_key, model=model_name),
            retriever=st.session_state.vector_store.as_retriever(),
            memory=memory
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
                    response = process_query(
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
                    st.error(f"Errore durante l'elaborazione: {str(e)}")

if __name__ == "__main__":
    main()