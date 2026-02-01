import yaml
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


API_KEY = os.getenv("CHATNOIR_API_KEY", "YOUR_API_KEY_HERE")
API_BASE_URL = "https://www.chatnoir.eu"

# 搜索配置
INDEX_NAME = "cw22"  # ClueWeb22
TOP_K = 10



def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _get_callbacks(source: str):
    try:
        from dashboard.prompt_logger import get_prompt_callbacks

        return get_prompt_callbacks(source)
    except Exception:
        return []


def get_llm_from_config(config_path='config.yaml'):
    config = load_config(config_path)
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'openai')
    model_name = llm_config.get('model', 'gpt-4o')
    temperature = llm_config.get('temperature', 0)
    callbacks = _get_callbacks("query_generator")

    if provider == 'openai':
        return ChatOpenAI(model=model_name, temperature=temperature, callbacks=callbacks)
    elif provider == 'anthropic':
        return ChatAnthropic(model=model_name, temperature=temperature, callbacks=callbacks)
    elif provider == 'gemini':
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, callbacks=callbacks)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
