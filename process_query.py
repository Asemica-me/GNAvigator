import asyncio

async def process_query(vector_store, client_data, system_prompt_template_str: str, user_question: str, chat_history: list):
    client, model_name = client_data

    # Step 1: Retrieve relevant pages
    retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # k can be tuned
    relevant_docs_from_vs = await retriever.ainvoke(user_question) # Langchain's ainvoke is async

    relevant_pages_urls = list(set([doc.metadata['url'] for doc in relevant_docs_from_vs if doc.metadata and 'url' in doc.metadata]))

    # Step 2: Fetch content in parallel
    # cached_fetch_wiki_content is already async and cached, which is good.
    if relevant_pages_urls:
        fetch_tasks = [cached_fetch_wiki_content(url) for url in relevant_pages_urls]
        page_contents_dicts = await asyncio.gather(*fetch_tasks)
    else:
        page_contents_dicts = []
        print("No relevant page URLs found from vector store.")

    # Step 3: Process and chunk content for context
    combined_context_parts = []
    for content_dict in filter(None, page_contents_dicts):
        if isinstance(content_dict, dict) and 'text' in content_dict:
            full_text = "\n\n".join(filter(None, [
                content_dict.get('text', ''),
                content_dict.get('tables', ''), # Tables can be large, consider summarizing or selective inclusion
                content_dict.get('images', '')  # Image markdown might not be ideal for LLM context unless specifically handled
            ]))
            if full_text.strip():
                # Chunk this fetched content for the prompt context
                # These chunks are different from the ones stored in the vector store
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150) # Tune for context window
                chunks = text_splitter.split_text(full_text)
                combined_context_parts.extend(chunks[:2]) # Take top 2 chunks per fetched page
        else:
            print(f"Warning: Fetched content has unexpected format: {content_dict}")


    # Limit total context to, e.g., 5 chunks to avoid overly long prompts
    MAX_CONTEXT_CHUNKS_FOR_PROMPT = 5
    context_str = "\n\n---\n\n".join(combined_context_parts[:MAX_CONTEXT_CHUNKS_FOR_PROMPT])

    if not context_str.strip() and relevant_docs_from_vs: # Some docs were found, but fetching/processing failed
        context_str = "No detailed content could be retrieved for the relevant documents. Please try rephrasing your question or ask about a general topic from the GNA manual."
    elif not context_str.strip(): # No docs found at all
        context_str = "I could not find specific information related to your query in the GNA manual. Please try asking in a different way."


    # Step 4: Prepare messages for Mistral API
    final_system_prompt = system_prompt_template_str.format(context=context_str)
    
    mistral_messages = [{"role": "system", "content": final_system_prompt}]
    for msg in chat_history[-3:]: # Keep last 3 turns for conversation context
        if isinstance(msg, HumanMessage):
            mistral_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            mistral_messages.append({"role": "assistant", "content": msg.content})
        # Add handling for your dict format if it's still in st.session_state.chat_history
        elif isinstance(msg, dict) and "role" in msg and "content" in msg: # From your older history format
             mistral_messages.append({"role": msg["role"], "content": msg["content"]})


    mistral_messages.append({"role": "user", "content": user_question})

    # Step 5: Generate response using the universal retry mechanism
    try:
        # Ensure you are using the correct method for your mistralai library version
        # For mistralai >= 1.0.0, it's client.chat.completions.create
        chat_completion = await invoke_mistral_api_with_retry(
            client.chat.completions.create, # Pass the function itself
            model=model_name,
            messages=mistral_messages,
            temperature=0.0, # As per original code
            max_tokens=2000  # As per original code
        )
        response_content = chat_completion.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"Error generating response in process_query: {e}")
        # import traceback # For debugging
        # print(traceback.format_exc())
        return "I'm sorry, but I encountered an issue while trying to generate a response. Please try again."

from async_lru import alru_cache
@alru_cache(maxsize=1000)
async def cached_fetch_wiki_content(url: str) -> dict:
    """Cache with 1-hour expiration and size limits"""
    content = await fetch_wiki_content(url)
    return content

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
        st.error(f"Error occurred: {str(e)}")
        print(f"Errore nella gestione della risposta: {e}") 