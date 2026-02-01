from pydantic import BaseModel, Field
from typing import List, Dict, Any

# ==========================
# Phase 1: Context Extraction
# ==========================
class KeywordCluster(BaseModel):
    core: List[str] = Field(..., description="The primary search terms.")
    lsi_synonyms: List[str] = Field(..., description="Semantically related terms.")
    keyphrases: List[str] = Field(..., description="Specific, lower-volume but high-intent queries.")

class PersonaProfile(BaseModel):
    name: str = Field(..., description="Specific label for the persona.")
    description: str = Field(..., description="Brief description of the persona's situation.")
    # pain_point: str = Field(..., description="What problem are they desperately trying to solve?")

class SearchProfile(BaseModel):
    """Phase 1 Output: Detailed analysis of keywords and personas"""
    keyword_cluster: KeywordCluster
    target_personas: List[PersonaProfile]

# ==========================
# Phase 2: LLM Output (Atomic Unit)
# ==========================
class IntentQueries(BaseModel):
    """LLM 在单次调用中返回的数据：针对 1个 Persona 的 4种 Intent"""
    navigational: List[str] = Field(..., description="5 queries looking for specific page/brand.")
    informational: List[str] = Field(..., description="5 queries seeking answers/guides (Question format).")
    commercial: List[str] = Field(..., description="5 queries comparing options/reviews.")
    transactional: List[str] = Field(..., description="5 queries for buying/downloading.")

class DeduplicationResult(BaseModel):
    """Result of the query deduplication process."""
    unique_queries: List[str] = Field(
        description="The final list of deduplicated, unique search queries."
    )

# ==========================
# Filter Models
# ==========================
class ApplicableIntents(BaseModel):
    """Filter 1 Output: Intent types suitable for this document"""
    intents: List[str] = Field(
        description="List of applicable intent types: navigational, informational, commercial, transactional"
    )
    reasoning: str = Field(description="Reasoning for the decision")

class DomainFilterResult(BaseModel):
    """Filter 2 Output: Domain relevance filtering result"""
    relevant_queries: List[str] = Field(description="Queries in the same domain as the document")
    filtered_queries: List[str] = Field(description="Queries filtered out due to completely different domain")

# ==========================
# Storage Models (用于保存到 JSON)
# ==========================
class PersonaVariation(BaseModel):
    """保存结构：特定 Persona 的 Query 集合"""
    target_persona: str
    queries: IntentQueries

class KeywordGroup(BaseModel):
    """保存结构：单个 Keyword 下的所有 Persona 变体"""
    focus_keyword: str
    variations: List[PersonaVariation]

class DatasetSplit(BaseModel):
    train: List[str]
    test: List[str]

class FinalOutput(BaseModel):
    uuid: str
    source_title: str
    keywords: KeywordCluster
    personas: List[PersonaProfile]
    # Detailed hierarchical structure (for analysis)
    detailed_data: List[KeywordGroup]
    # Flat dataset (for training)
    dataset: DatasetSplit
    stats: Dict[str, Any]
