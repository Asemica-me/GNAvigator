import os
import traceback
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

#from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
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
        doc.metadata["chunk_id"] = generate_chunk_id(url, i)

        final_content = f"{title}\n{description}\n{doc.page_content}"
        doc.page_content = final_content
        formatted_chunks.append(doc)

        # # Debug: visualizzare il primo chunk
        # if i == 0:
        #     print(f"\nChunk {i + 1}:")
        #     print(f"Title: {title}")
        #     print(f"Description: {description}")
        #     print(f"Content: {content}")
        #     print(f"URL: {url}")
        #     print(f"chunk_id: {chunk_id}")

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
                metadata=doc.metadata  # Keep existing metadata including chunk_id
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
    chat_client = ChatMistralAI(api_key=api_key, model=model_name)
    print("Mistral configurato correttamente")
    return (client, rate_limiter, model_name, chat_client)

def invoke_with_retry(client, rate_limiter, model_name, messages, 
                     max_retries=5, initial_delay=15.0, backoff_factor=2.5):
    """Retry logic with rate limit handling"""
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
    mistral_client = client_data
    client = mistral_client["client"]
    rate_limiter = mistral_client["rate_limiter"]
    model_name = mistral_client["model_name"]
    chat_client = mistral_client["chat_client"]

    # Rate limit before making the API call
    rate_limiter.consume()
    
    # Ricerca documenti rilevanti
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(user_question)
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
    #response = invoke_with_retry(client, rate_limiter, model_name, messages)
    response = st.session_state.conversation.invoke(
        {"input": user_question},  # Use the correct key here
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    # Extract and format URLs from source documents
    urls = [doc.metadata.get('url', '') for doc in docs]
    unique_urls = list(set(filter(None, urls)))  # Remove duplicates and empty strings

    if unique_urls:
        response += f"\n\nFonti: {', '.join(unique_urls)}"  # "Sources" in Italian
    else:
        response  # "No sources available"
    
    
    return response

def handle_style_and_responses(user_question: str) -> None:
    try:
        # Chiamata corretta alla catena
        response = st.session_state.conversation.invoke(
            {"question": user_question},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )

        # Estrazione risposta
        answer = response.content if hasattr(response, 'content') else response

        # Estrazione fonti
        if hasattr(response, 'context'):
            sources = list(set(
                doc.metadata.get("url", "") 
                for doc in response.context 
                if doc.metadata.get("url")
            ))
            answer += f"\n\nFonti: {', '.join(sources)}" if sources else ""

        # Aggiornamento cronologia
        st.session_state.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=answer)
        ])

        # Display styling
        human_style = (
            "background-color: #3f444f; "
            "border-radius: 10px; "
            "padding: 10px; "
            "margin: 5px 0;"
        )
        
        bot_style = (
            "background-color: #2d3136; "
            "border-radius: 10px; "
            "padding: 10px; "
            "margin: 5px 0;"
        )

        # Render messages
        for i, message in enumerate(st.session_state.chat_history):
            if isinstance(message, HumanMessage):
                st.markdown(
                    f"<div style='text-align: right; margin: 10px 0;'>"
                    f"<div style='{human_style}'>"
                    f"<b>Utente</b><br>{message.content}"
                    f"</div></div>", 
                    unsafe_allow_html=True
                )
            elif isinstance(message, AIMessage):
                st.markdown(
                    f"<div style='text-align: left; margin: 10px 0;'>"
                    f"<div style='{bot_style}'>"
                    f"<b>Assistente AI</b><br>{message.content}"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

        # Auto-scroll to bottom
        components.html(
            """
            <script>
                window.parent.document.querySelector('.main').scrollTo({
                    top: window.parent.document.querySelector('.main').scrollHeight,
                    behavior: 'smooth'
                });
            </script>
            """,
            height=0
        )

    except Exception as e:
        st.error(f"Si è verificato un errore: {str(e)}")
        print(f"Errore dettagliato: {traceback.format_exc()}") 



def main():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("MODEL")
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    system_prompt = """
        Sei un'assistente esperto del [Geoportale Nazionale dell’Archeologia (GNA)](https://gna.cultura.gov.it/wiki/index.php/Pagina_principale), gestito dall'Istituto centrale per il catalogo e la documentazione (ICCD).

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
        page_title="Assistente virtuale GNA COPIA",
        page_icon=":bust_in_silhouette:"
    )

    # Inizializzazione componenti
    if "mistral_client" not in st.session_state:
        client, rate_limiter, model_name, chat_client = create_mistral_client(api_key, model_name)
        st.session_state.mistral_client = {
            "client": client,
            "rate_limiter": rate_limiter,
            "model_name": model_name,
            "chat_client": chat_client
        }
    
    if "vector_store" not in st.session_state:
        dataset = load_dataset("gna_kg_dataset.csv")
        chunks = create_chunks(dataset, chunk_size=1000, chunk_overlap=200)  # Pass required parameters
        st.session_state.vector_store = create_or_get_vector_store(chunks, api_key)

    retriever = st.session_state.vector_store.as_retriever()
    
    if "conversation" not in st.session_state:
        mistral_client = st.session_state.mistral_client

        # Definizione del prompt corretto
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Creazione della catena corretta
        chain = (
            {"question": RunnablePassthrough(), "context": retriever} 
            | prompt 
            | mistral_client["chat_client"]
        )

        # Configurazione della cronologia
        def get_session_history(session_id: str) -> ChatMessageHistory:
            if "session_history" not in st.session_state:
                st.session_state.session_history = {}
            return st.session_state.session_history.setdefault(session_id, ChatMessageHistory())

        # Creazione conversazione finale
        st.session_state.conversation = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
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
                process_query(
                    st.session_state.vector_store,
                    st.session_state.mistral_client,
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