import time

from chunk_gatherer.chunk_gatherer import ChunkGatherer
from tavily import TavilyClient


class WebGatherer(ChunkGatherer):
    def __init__(self, api_key: str, delay_between_requests: float = 1.0, max_retries: int = 3):
        """
        Initialize Web Gatherer with rate limiting and retry logic.

        Args:
            api_key: Tavily API key
            delay_between_requests: Seconds to wait between requests to avoid rate limits
            max_retries: Maximum number of retry attempts on rate limit errors
        """
        self.client = TavilyClient(api_key=api_key)
        self.delay = delay_between_requests
        self.max_retries = max_retries
        self._last_request_time: float = 0.0

    def gather(self, key: str) -> str:
        return ""
        """
        Gather content from URL with rate limiting and exponential backoff retry.

        Args:
            key: URL to extract content from

        Returns:
            Extracted content string
        """
        for attempt in range(self.max_retries):
            # Rate limiting: wait if needed
            elapsed = time.time() - self._last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

            try:
                response = self.client.extract(urls=[key])
                self._last_request_time = time.time()
                return response.get("content", "")

            except Exception as e:
                self._last_request_time = time.time()
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                if "rate" in error_msg or "excessive" in error_msg or "blocked" in error_msg:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: wait 2^attempt seconds
                        wait_time = 2**attempt
                        print(
                            f"Rate limit hit for {key}, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Max retries reached for {key}, giving up.")
                        raise
                else:
                    # Not a rate limit error, raise immediately
                    raise

        return ""
