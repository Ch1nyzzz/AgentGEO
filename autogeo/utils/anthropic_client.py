"""
Anthropic API client for AutoGEO.
"""
import os
import time
from dotenv import load_dotenv
import anthropic
from .constants import COMMON_SYSTEM_PROMPT, MAX_RETRIES, RETRY_DELAY_SECONDS

load_dotenv("keys.env")
load_dotenv()  # Also load from .env

# Initialize Anthropic client
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def call_anthropic(
    user_prompt: str,
    system_prompt: str = COMMON_SYSTEM_PROMPT,
    model_name: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.7
) -> str:
    """Call Anthropic API with retry logic.

    Args:
        user_prompt: User prompt text
        system_prompt: System prompt text (default: COMMON_SYSTEM_PROMPT)
        model_name: Anthropic model name (default: "claude-haiku-4-5-20251001")
        temperature: Sampling temperature (default: 0.7)

    Returns:
        Response text from Anthropic

    Raises:
        Exception: If API call fails after all retries
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=8192,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.content[0].text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise
