import time
from threading import Lock

# TokenBucketRateLimiter class (copied from your code)
class TokenBucketRateLimiter:
    def __init__(self, rate, capacity):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # maximum bucket capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
        print(f"Refilled tokens: {self.tokens:.2f}")

    def consume(self, tokens=1):
        while True:
            with self.lock:
                self._refill()
                print(f"Available tokens: {self.tokens:.2f}")
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    print(f"Consumed 1 token. Remaining tokens: {self.tokens:.2f}")
                    return
            time.sleep(0.01)

# Test the TokenBucketRateLimiter
def test_rate_limiter():
    # Create a rate limiter with 1 request every 2 seconds and a capacity of 5
    rate_limiter = TokenBucketRateLimiter(rate=0.5, capacity=5)

    print("Starting rate limiter test...")
    for i in range(10):  # Simulate 10 requests
        print(f"Request {i + 1}: ", end="")
        try:
            rate_limiter.consume()  # Consume a token for each request
            print("Allowed")
        except Exception as e:
            print(f"Blocked: {e}")
        time.sleep(0.5)  # Simulate a delay between requests

if __name__ == "__main__":
    test_rate_limiter()