import os
from enum import Enum
from typing import Optional

import yaml
from dotenv import load_dotenv


load_dotenv()


class LLMTask(str, Enum):
    """LLM 任务类型枚举"""
    GENERATION = "generation"          # 答案生成
    CITATION_CHECK = "citation_check"  # 引用检查
    DIAGNOSIS = "diagnosis"            # 诊断分析
    TOOL_STRATEGY = "tool_strategy"    # 工具策略选择
    GEO_SCORE = "geo_score"            # GEO 评分评估

os.environ.setdefault("KMP_USE_SHM", "0")

API_KEY = os.getenv("CHATNOIR_API_KEY", "YOUR_API_KEY_HERE")
API_BASE_URL = "https://www.chatnoir.eu"

# 搜索配置
INDEX_NAME = "cw22"  # ClueWeb22
TOP_K = 10



def load_config(config_path='config.yaml'):
    from pathlib import Path

    # 尝试直接打开
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # 尝试从 geo_agent 目录加载
    geo_agent_dir = Path(__file__).parent
    alt_path = geo_agent_dir / Path(config_path).name
    if alt_path.exists():
        with open(alt_path, 'r') as f:
            return yaml.safe_load(f)

    # 尝试从项目根目录加载
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
    兼容 LangChain 风格输入：
    - str
    - List[Tuple[role, content]]，role 可为 system/human/user/ai/assistant
    - 具有 `.content`/`.type` 的 message 对象（尽力适配）
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
    """OpenAI SDK 封装，提供 `.invoke`/`.ainvoke` 接口。"""

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
    """Anthropic SDK 封装，提供 `.invoke`/`.ainvoke` 接口。"""

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
    """Gemini SDK 封装，提供 `.invoke`/`.ainvoke` 接口（简单拼接消息）。"""

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
        # 旧版 google.generativeai 多数为同步实现；这里用线程封装。
        import asyncio

        return await asyncio.to_thread(self.invoke, prompt_input)


def get_llm_from_config(config_path='config.yaml'):
    """从 config.yaml 读取 LLM 配置并返回可 `.invoke`/`.ainvoke` 的客户端。"""
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
    获取特定任务的 LLM 客户端

    优先级：llm_tasks.{task} > llm（全局默认）

    Args:
        config_path: 配置文件路径
        task: LLM 任务类型，为 None 时使用全局默认配置

    Returns:
        LLM 客户端实例
    """
    config = load_config(config_path)

    # 尝试获取任务特定配置
    task_config = None
    if task:
        llm_tasks = config.get('llm_tasks', {})
        task_config = llm_tasks.get(task.value)

    # 回退到全局配置
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
