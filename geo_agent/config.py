import os
from enum import Enum
from typing import Optional

import yaml
from dotenv import load_dotenv


load_dotenv()


class LLMTask(str, Enum):
    """LLM task type enumeration"""
    GENERATION = "generation"          # Answer generation
    CITATION_CHECK = "citation_check"  # Citation checking
    DIAGNOSIS = "diagnosis"            # Diagnostic analysis
    TOOL_STRATEGY = "tool_strategy"    # Tool strategy selection
    GEO_SCORE = "geo_score"            # GEO score evaluation

os.environ.setdefault("KMP_USE_SHM", "0")

API_KEY = os.getenv("CHATNOIR_API_KEY")
if not API_KEY:
    import warnings
    warnings.warn("CHATNOIR_API_KEY environment variable not set. ChatNoir search will not work.")
API_BASE_URL = "https://www.chatnoir.eu"

# Search configuration
INDEX_NAME = "cw22"  # ClueWeb22
TOP_K = 10



def load_config(config_path='config.yaml'):
    from pathlib import Path

    # Try to open directly
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Try to load from geo_agent directory
    geo_agent_dir = Path(__file__).parent
    alt_path = geo_agent_dir / Path(config_path).name
    if alt_path.exists():
        with open(alt_path, 'r') as f:
            return yaml.safe_load(f)

    # Try to load from project root directory
    repo_root = geo_agent_dir.parent
    alt_path2 = repo_root / config_path
    if alt_path2.exists():
        with open(alt_path2, 'r') as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(f"Config file not found: {config_path}")

def _get_callbacks(source: str):
    try:
        from dashboard.prompt_logger import get_prompt_callbacks

        return get_prompt_callbacks(source)
    except Exception:
        return []


class LLMResponse:
    """Minimal response shim compatible with existing `.content` usage."""

    def __init__(self, content: str):
        self.content = content


def _normalize_chat_messages(prompt_input):
    """
    Compatible with LangChain-style input:
    - str
    - List[Tuple[role, content]], role can be system/human/user/ai/assistant
    - Message objects with `.content`/`.type` attributes (best effort adaptation)
    """
    if isinstance(prompt_input, str):
        return [{"role": "user", "content": prompt_input}]

    if isinstance(prompt_input, list):
        normalized = []
        for item in prompt_input:
            role = None
            content = None

            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
            elif isinstance(item, dict):
                role = item.get("role") or item.get("type")
                content = item.get("content")
            else:
                role = getattr(item, "type", None) or getattr(item, "role", None)
                content = getattr(item, "content", None)

            if content is None:
                content = str(item)

            role = (role or "user").lower()
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            normalized.append({"role": role, "content": str(content)})
        return normalized

    return [{"role": "user", "content": str(prompt_input)}]


class OpenAIChatLLM:
    """OpenAI SDK wrapper providing `.invoke`/`.ainvoke` interface."""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt_input):
        from openai import OpenAI

        client = OpenAI()
        messages = _normalize_chat_messages(prompt_input)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        content = (resp.choices[0].message.content or "").strip()
        return LLMResponse(content)

    async def ainvoke(self, prompt_input):
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        messages = _normalize_chat_messages(prompt_input)
        resp = await client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        content = (resp.choices[0].message.content or "").strip()
        return LLMResponse(content)


class AnthropicChatLLM:
    """Anthropic SDK wrapper providing `.invoke`/`.ainvoke` interface."""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def _split_system(self, messages):
        system_parts = []
        non_system = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                role = msg["role"]
                if role not in {"user", "assistant"}:
                    role = "user"
                non_system.append({"role": role, "content": msg["content"]})
        return ("\n\n".join(system_parts).strip() or None), non_system

    def invoke(self, prompt_input):
        from anthropic import Anthropic

        client = Anthropic()
        messages = _normalize_chat_messages(prompt_input)
        system, non_system = self._split_system(messages)
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": non_system,
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        content = "".join([block.text for block in resp.content if getattr(block, "type", "") == "text"]).strip()
        return LLMResponse(content)

    async def ainvoke(self, prompt_input):
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        messages = _normalize_chat_messages(prompt_input)
        system, non_system = self._split_system(messages)
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": non_system,
            "max_tokens": 4096,
        }
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
        content = "".join([block.text for block in resp.content if getattr(block, "type", "") == "text"]).strip()
        return LLMResponse(content)


class GeminiChatLLM:
    """Gemini SDK wrapper providing `.invoke`/`.ainvoke` interface (simple message concatenation)."""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def _flatten(self, messages):
        parts = []
        for msg in messages:
            parts.append(f"[{msg['role']}] {msg['content']}")
        return "\n\n".join(parts)

    def invoke(self, prompt_input):
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model)
        prompt = self._flatten(_normalize_chat_messages(prompt_input))
        resp = model.generate_content(prompt, generation_config={"temperature": self.temperature})
        return LLMResponse((getattr(resp, "text", "") or "").strip())

    async def ainvoke(self, prompt_input):
        # Older google.generativeai versions are mostly synchronous; wrapping with thread here.
        import asyncio

        return await asyncio.to_thread(self.invoke, prompt_input)


def get_llm_from_config(config_path='config.yaml'):
    """Read LLM configuration from config.yaml and return a client with `.invoke`/`.ainvoke` interface."""
    config = load_config(config_path)
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'openai')
    model_name = llm_config.get('model', 'gpt-4.1-mini')
    temperature = llm_config.get('temperature', 0)

    if provider == 'openai':
        return OpenAIChatLLM(model=model_name, temperature=temperature)
    elif provider == 'anthropic':
        return AnthropicChatLLM(model=model_name, temperature=temperature)
    elif provider == 'gemini':
        return GeminiChatLLM(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_for_task(config_path: str = 'config.yaml', task: Optional[LLMTask] = None):
    """
    Get LLM client for a specific task

    Priority: llm_tasks.{task} > llm (global default)

    Args:
        config_path: Configuration file path
        task: LLM task type, uses global default configuration when None

    Returns:
        LLM client instance
    """
    config = load_config(config_path)

    # Try to get task-specific configuration
    task_config = None
    if task:
        llm_tasks = config.get('llm_tasks', {})
        task_config = llm_tasks.get(task.value)

    # Fall back to global configuration
    if not task_config:
        task_config = config.get('llm', {})

    provider = task_config.get('provider', 'openai')
    model = task_config.get('model', 'gpt-4.1-mini')
    temperature = task_config.get('temperature', 0)

    if provider == 'openai':
        return OpenAIChatLLM(model=model, temperature=temperature)
    elif provider == 'anthropic':
        return AnthropicChatLLM(model=model, temperature=temperature)
    elif provider == 'gemini':
        return GeminiChatLLM(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_search_client_from_config(config_path='config.yaml'):
    config = load_config(config_path)
    search_config = config.get('search', {})
    provider = search_config.get('provider', 'tavily')
    
    if provider == 'chatnoir':
        from geo_agent.search_engine.chatnoir import ChatNoirClient
        max_results = search_config.get('max_results', 14)
        return ChatNoirClient(max_results=max_results)
    elif provider == 'tavily':
        from geo_agent.search_engine.tavily import TavilyClient
        max_results = search_config.get('max_results', 10)
        return TavilyClient(max_results=max_results)
    else:
        raise ValueError(f"Unsupported search provider: {provider}")

def get_generator_from_config(config_path='config.yaml'):
    config = load_config(config_path)
    gen_config = config.get('generator', {})
    
    generator_type = gen_config.get('method', gen_config.get('provider', 'in-context'))

    if generator_type == 'in-context':
        from geo_agent.generate import InContextGenerator
        return InContextGenerator(config_path)
    elif generator_type == 'attr_evaluator':
        from geo_agent.generate import AttrFirstThenGenerate
        return AttrFirstThenGenerate(config_path)
    else:
        raise ValueError(f"Unsupported generator type: {generator_type}")
