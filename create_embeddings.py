import asyncio
import httpx
from httpx import Limits

MISTRAL_REQUESTS_PER_SECOND = 1
API_RATE_LIMITER_CAPACITY = MISTRAL_REQUESTS_PER_SECOND * 2

# Classe per la gestione del rate limiting
class AsyncTokenBucketRateLimiter:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens (requests) per second
        self.capacity = float(capacity)
        self.tokens = float(capacity) # Start full
        self.last_refill = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1):
        if tokens > self.capacity:
            raise ValueError("Cannot consume more tokens than capacity in a single request.")

        while True:
            async with self.lock:
                now = asyncio.get_event_loop().time()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return # Token consumed

            # If not enough tokens, calculate wait time
            # Re-check under lock to avoid race condition for wait_time calculation
            async with self.lock:
                needed_tokens = tokens - self.tokens # How many more tokens are needed
                current_wait_time = 0
                if needed_tokens > 0:
                    current_wait_time = needed_tokens / self.rate
            
            if current_wait_time > 0:
                await asyncio.sleep(current_wait_time)
            else: # Should have enough tokens now, but add a tiny sleep to yield control if loop is too tight
                await asyncio.sleep(0.001) # Minimal sleep to allow other tasks

api_rate_limiter = AsyncTokenBucketRateLimiter(rate=MISTRAL_REQUESTS_PER_SECOND, capacity=API_RATE_LIMITER_CAPACITY)

async def invoke_mistral_api_with_retry(
    api_call_func, # The actual async API call function (e.g., client.chat.completions.create)
    *args,
    max_retries: int = 5,
    initial_delay: float = 2.0, # Start with a slightly longer delay for 429
    backoff_factor: float = 2.0,
    **kwargs
):
    """
    Generic wrapper for Mistral API calls with rate limiting and retries.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        await api_rate_limiter.consume(1) # Wait for token before attempting call
        try:
            response = await api_call_func(*args, **kwargs)
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after_header = e.response.headers.get('Retry-After')
                wait_time = delay
                if retry_after_header:
                    try:
                        wait_time = float(retry_after_header)
                    except ValueError:
                        print(f"Warning: Could not parse Retry-After header: {retry_after_header}. Using default backoff.")
                
                # Respect server's request, but also implement progressive backoff
                effective_wait_time = max(wait_time, delay)
                print(f"Rate limited (429). Retrying in {effective_wait_time:.2f} seconds (attempt {attempt + 1}/{max_retries}). Response: {e.response.text}")
                await asyncio.sleep(effective_wait_time)
                delay *= backoff_factor
            elif e.response.status_code >= 500: # Server-side errors
                print(f"Mistral API server error ({e.response.status_code}). Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries}). Error: {e}")
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else: # Other client-side HTTP errors
                print(f"Mistral API HTTP error: {e}. Response: {e.response.text}")
                raise
        except httpx.RequestError as e: # Network errors (ConnectTimeout, ReadTimeout etc.)
            print(f"Mistral API request error: {e}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
            delay *= backoff_factor
        except Exception as e:
            print(f"Unexpected error during Mistral API call: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= backoff_factor
    raise Exception(f"Max retries ({max_retries}) exceeded for Mistral API call.")

async def get_embeddings_in_batches(client, texts: list[str], embedding_model_name: str, batch_size: int = 32): # Check Mistral's recommended batch size
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Each batch is one API call
            response = await invoke_mistral_api_with_retry(
                client.embeddings.create, # Pass the function itself
                model=embedding_model_name,
                input=batch_texts
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"Failed to get embeddings for batch starting at index {i}: {e}")
            # Decide on error handling: skip batch, raise error, or return partial results
            # For simplicity, re-raising here.
            raise
    return all_embeddings