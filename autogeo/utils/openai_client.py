"""
OpenAI API client for AutoGEO.
"""
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from .constants import COMMON_SYSTEM_PROMPT, MAX_RETRIES, RETRY_DELAY_SECONDS

load_dotenv("keys.env")

# Initialize OpenAI client
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def call_openai(
    user_prompt: str,
    system_prompt: str = COMMON_SYSTEM_PROMPT,
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.7
) -> str:
    """Call OpenAI API with retry logic.

    Args:
        user_prompt: User prompt text
        system_prompt: System prompt text (default: COMMON_SYSTEM_PROMPT)
        model_name: OpenAI model name (default: "gpt-4.1-mini")
        temperature: Sampling temperature (default: 0.7)

    Returns:
        Response text from OpenAI

    Raises:
        Exception: If API call fails after all retries
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise
