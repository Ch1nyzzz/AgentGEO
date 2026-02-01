from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class WebPage(BaseModel):
    url: str
    raw_html: str
    cleaned_content: str
    title: str = ""

class SearchResult(BaseModel):
    idx: int
    title: str
    snippet: str
    url: str
    uuid: str = ""  # 可选，兼容旧缓存
    raw_content: str = ""  # Tavily raw_content (Markdown 格式)

class CitationCheckResult(BaseModel):
    is_cited: bool
    generated_answer: str
    citations_found_idx: List[int]
    # GEO Score（可选字段，用于评测）
    geo_score: Optional[Any] = None

class AnalysisResult(BaseModel):
    """LLM analysis of reasons for non-citation and next action"""
    reasoning: str = Field(..., description="Analysis of why the target document was not cited (e.g., missing key information, poor structure, lack of authority)")
    selected_tool_name: str = Field(..., description="Name of the tool selected for optimization")
    tool_arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool, must conform to the tool's schema")
    target_chunk_index: Optional[int] = Field(None, description="Index of the chunk to modify (only used when document is chunked)")

class OptimizationState(BaseModel):
    """Track the state of the optimization process"""
    query: str
    current_content: str
    iteration: int
    history: List[str] = [] # Record past operations to prevent infinite loops