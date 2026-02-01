import uuid
import time
from typing import List
from bs4 import BeautifulSoup
import json
from sklearn.model_selection import train_test_split
from langchain_core.prompts import ChatPromptTemplate

from .config import get_llm_from_config
from .models import (
    SearchProfile, IntentQueries, DeduplicationResult, PersonaProfile,
    KeywordGroup, PersonaVariation, FinalOutput, DatasetSplit,
    ApplicableIntents, DomainFilterResult
)

class SEOQueryPipeline:
    def __init__(self, config_path: str = 'config.yaml'):
        self.llm = get_llm_from_config(config_path)

    def _extract_title(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        return soup.title.string.strip() if soup.title else "Untitled"

    def _extract_summary(self, html: str, max_chars: int = 8000) -> str:
        """Extract document summary (first N characters of text content)"""
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text[:max_chars] if text else ""

    # ----------------------------------------------------------------
    # Phase 1: Profile Generation
    # ----------------------------------------------------------------

    def generate_profile(self, title: str, doc_summary: str) -> SearchProfile:
        print(" -> Generating Profile (Keywords & Personas)...")

        system_prompt = """
        You are a Senior Semantic SEO Specialist and User Intent Analyst. Your goal is to reverse-engineer the search intent behind a specific webpage title to uncover high-value keyword clusters and precise audience segments.

        # Task
        Analyze the title to identify its core topics and assess the types of users who are likely to be interested in those topics. Crucially, you must expand these topics to include:

        # Analysis Requirements
        1.  Topic/Keyword Extraction:
            * Core Keywords: The primary search terms.
            * LSI/Synonyms: Synonyms and semantically related terms (Latent Semantic Indexing) that add context.
            * Keyword Phrases: A specific, multi-word search query that reflects clear user intent.
            * Ambiguity: If a keyword carries multiple meanings (polysemy), appropriately expand it by adding necessary modifiers to strictly define its specific meaning based on the web content. (e.g., clarify "Jaguar" to "Jaguar Car" or "Jaguar Animal").

        2.  Persona Profiling:
            * Identify distinct user personas
            * For each persona, identify:
                * Role/Demographic: Specific groups (e.g., Infants, Children, Teens, Adults, Seniors, Pregnant women).
                * Description: A concise overview of the persona's circumstances, offering general context without referencing the keyword specifically.

        # Output Format
        Please present your analysis in the following JSON format:

        {{
            "keyword_cluster": {{
            "core": ["keyword1", "keyword2"],
            "lsi_synonyms": ["synonym1", "synonym2"],
            "keyphrases": ["phrase1", "phrase2"]
            }},
            "target_personas": [
            {{
                "name": "Persona Name",
                "description": "Persona Description"
            }}
            ]
        }}
        """

        structured_llm = self.llm.with_structured_output(SearchProfile)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Title: {title}\n\nDocument Summary: {doc_summary}")
        ])

        chain = prompt | structured_llm
        return chain.invoke({"title": title, "doc_summary": doc_summary})

    # ----------------------------------------------------------------
    # Filter 1: Determine Applicable Intents
    # ----------------------------------------------------------------
    def determine_applicable_intents(self, title: str, doc_summary: str) -> ApplicableIntents:
        """Determine which search intent types are suitable for this document"""
        print(" -> Determining applicable intents...")

        system_prompt = """
        You are a Senior SEO Specialist and Google Search Quality Rater. Your task is to analyze a website's **Title** and **Content Summary** to determine which User Search Intents this page satisfies.

        # The 4 Search Intents (Definitions)
        You must select from the following four categories based on the user's goal:

        1.  **Informational (KNOW):** The user wants to learn something, find an answer, or solve a problem. (e.g., Guides, Tutorials, Definitions).
        2.  **Commercial Investigation (COMPARE):** The user is in the consideration phase, researching products/services but not ready to pay yet. (e.g., Reviews, Comparisons, "Best of" lists).
        3.  **Transactional (DO):** The user is ready to take a specific action, usually involving a conversion. (e.g., Buy, Subscribe, Download, Register).
        4.  **Navigational (GO):** The user is looking for a specific website or page. (e.g., Login pages, Contact pages, About Us).

        # Task Instructions
        1.  Analyze the provided **Title** and **Summary**.
        2.  Evaluate the page against **ALL 4 intents**.
        3.  **Filter Logic (Crucial):** Only assign an intent if the content provides a **substantial and satisfying solution** for that intent.
            * *Example:* A blog post with a small "buy now" link in the footer is **NOT** Transactional. It is Informational.
            * *Example:* A product page with a short description is primarily Transactional, but NOT Informational (since it lacks depth).
        4.  **Output:** Return the applicable intents (can be multiple) with a confidence score and reasoning.

        # Output Format
        List the suitable intents and provide a short reasoning based on why they were NOT excluded.

        **Output Format:**
        - Intents: [List of intents chose from Informational, Commercial Investigation, Transactional, Navigational]
        - Reasoning: [Brief reasoning explaining the primary goal of the content and any secondary goals based on specific keywords or content features.]
        """

        structured_llm = self.llm.with_structured_output(ApplicableIntents)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Title: {title}\n\nDocument Summary:\n{doc_summary}")
        ])

        chain = prompt | structured_llm
        result = chain.invoke({"title": title, "doc_summary": doc_summary})
        print(f"   Applicable intents: {result.intents}")
        return result

    # ----------------------------------------------------------------
    # Phase 2: Atomic Query Generation (Single Persona)
    # ----------------------------------------------------------------
    def generate_queries_for_persona(
        self, keyword: str, persona: PersonaProfile,
        applicable_intents: List[str] = None
    ) -> IntentQueries:
        """
        Generate queries for a single Keyword and Persona combination.
        Only generates queries for applicable intent types.
        """
        if applicable_intents is None:
            applicable_intents = ["Navigational", "Informational", "Commercial Investigation", "Transactional"]

        # Build intent instructions dynamically
        intent_instructions = []
        if "Navigational" in applicable_intents:
            intent_instructions.append("""1. **Navigational (Go):**
            * The user is looking for a specific page (e.g., "pedigree foundation")""")
        if "Informational" in applicable_intents:
            intent_instructions.append("""2. **Informational (Know):**
            * The user is looking for general information on a topic (e.g., "can dogs eat spicy food").
            * *Constraint:* Match the user's vocabulary level (e.g., simple terms for beginners vs. jargon for experts).""")
        if "Commercial Investigation" in applicable_intents:
            intent_instructions.append("""3. **Commercial Investigation (Investigate):**
            * The user is researching their options before making the final decision on which product to buy (e.g., "best dry dog food")""")
        if "Transactional" in applicable_intents:
            intent_instructions.append("""4. **Transactional (Do):**
            * The user is looking for a specific product or brand, with the intention to make a purchase (e.g., "pedigree puppy food")""")

        intent_section = "\n".join(intent_instructions)

        system_prompt = f"""# Role
        You are a Realistic Search Engine User Simulator. Your goal is to mimic the exact search behavior, vocabulary, and intent of a specific user segment.

        # Input Context
        * **Focus Keyword:** {{keyword}}
        * **User Persona Profile:**
            * **Role/Demographics:** {{persona_name}} (e.g., Senior Citizen, Python Developer, Busy Mom)
            * **User Description:** {{persona_description}} (Current situation, experience level, environment)

        # Task
        Generate exactly 5 search queries related to "{{keyword}}" for EACH of the following applicable search intents.

        # Applicable Search Intent Categories & Guidelines
        {intent_section}

        # Output Format
        Return a JSON object with arrays for each intent type.
        For intent types NOT listed above, return an empty array [].

        {{{{
            "navigational": ["query1", ...] or [],
            "informational": ["query1", ...] or [],
            "commercial": ["query1", ...] or [],
            "transactional": ["query1", ...] or []
        }}}}
        """

        structured_llm = self.llm.with_structured_output(IntentQueries)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Generate queries now.")
        ])

        chain = prompt | structured_llm
        result = chain.invoke({
            "keyword": keyword,
            "persona_name": persona.name,
            "persona_description": persona.description,
        })

        # Ensure non-applicable intents return empty lists
        if "Navigational" not in applicable_intents:
            result.navigational = []
        if "Informational" not in applicable_intents:
            result.informational = []
        if "Commercial Investigation" not in applicable_intents:
            result.commercial = []
        if "Transactional" not in applicable_intents:
            result.transactional = []

        return result


    def deduplicate_queries(self, queries: List[str]) -> DeduplicationResult:
        """
        deduplicate and clean queries
        """
        system_prompt = """You are an expert Semantic Query Optimizer and Data Cleaner. Your task is to analyze a list of user search queries and remove duplicates based on **User Intent** and **Semantic Equivalence**.

        ### Core Objective
        Reduce redundancy in the provided list while preserving every distinct user intent, specific entity (products/dates), and action type.

        ### Deduplication Logic (Step-by-Step)

        1.  **Exact & Morphological Duplicates:**
            * Remove queries that are identical strings (case-insensitive).
            * Remove queries that are merely reordered variations of keywords without changing the meaning (e.g., "interior 2021 g-wagon" vs. "2021 g-wagon interior").

        2.  **Semantic Equivalence (Synonyms):**
            * Merge queries where keywords are swappable synonyms in the specific context.
            * *Example:* "Luxury cabin" ≈ "Premium interior".
            * *Example:* "Specs" ≈ "Specifications" ≈ "Features".
            * *Example:* "Buy" ≈ "Purchase".

        3.  **Representative Selection (The "Canonical" Query):**
            * When a group of similar queries is identified, select **one** representative query to keep.
            * **Priority for selection:**
                1.  The most grammatically natural and fluent option.
                2.  The most specific/descriptive option (provided it covers the intent of the others).
                3.  The option that uses standard industry terminology.

        ### Strict Exclusion Rules (Do NOT Merge)

        1.  **Distinct Entities:** Do not merge queries referring to different models, versions, or years (e.g., "2021 model" is NOT the same as "2023 model"; "G-Class" is NOT the same as "GLE").
        2.  **Distinct Intents:** Do not merge different stages of the user journey.
            * *Transactional:* "Buy..."
            * *Informational:* "Reviews of...", "Specs of..."
            * *Navigational:* "Official site..."
            * *Comparison:* "Model A vs Model B"

        ### Output Format
        Return only the final deduplicated list of strings in valid JSON format: `["query1", "query2", ...]`."""

        queries = list(set([q.strip().lower() for q in queries]))
        structured_llm = self.llm.with_structured_output(DeduplicationResult)
        query_str = json.dumps(queries, indent=2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "The list of queries to deduplicate:\n{query_str}")
        ])

        chain = prompt | structured_llm
        return chain.invoke({
            "query_str": query_str
        })

    # ----------------------------------------------------------------
    # Filter 2: Domain Relevance Filter
    # ----------------------------------------------------------------
    def filter_by_domain(self, doc_summary: str, queries: List[str]) -> DomainFilterResult:
        """Filter out queries from completely different domains"""
        print(" -> Filtering by domain relevance...")

        system_prompt = """
        You are an SEO Relevance Expert and Topic Cluster Specialist. Your goal is to filter a list of search queries based on their potential alignment with a specific document.

        ### The Core Objective
        You must act as a **Broad Filter**. Your job is NOT to find only the "perfect" matches. Your job is to **remove noise**.
        * **Default Action:** KEEP the query.
        * **Exception:** REMOVE only if the query is definitely irrelevant.

        ### 1. Retention Rules (When to KEEP)
        * **Same Domain:** Keep all queries related to the document's general industry or topic (e.g., if Doc is about "SEO", keep "digital marketing" queries).
        * **Low Relevance / Tangential:** Keep queries that are loosely related. If we could add a paragraph to the document to answer this query, KEEP IT.
        * **Broad/Vague:** Keep broad queries (e.g., "what is marketing") if the document covers a specific aspect of that topic.
        * **Competitor Comparisons:** Keep queries comparing the subject to alternatives.

        ### 2. Exclusion Rules (When to REMOVE)
        Remove a query **strictly** if it meets one of these conditions:

        * **Semantic Mismatch (Polysemy):** The query uses a word that appears in the document but refers to a completely different concept/industry.
            * *Example (Syndication):* Document is about "Media Content Syndication".
                * **KEEP:** "content distribution platforms" (Same domain).
                * **REMOVE:** "bank loan syndication process" (Finance domain - completely different audience).
            * *Example (Python):* Document is about "Python Programming".
                * **REMOVE:** "types of python snakes in florida" (Zoology domain).

        * **Wrong Industry/Intent:** The user searching for this has **zero** overlap with the target audience of the document.
            * *Example:* Doc is about "Enterprise CRM Software"; Query is about "Free Minecraft skins" → REMOVE.

        ### 3. Decision Heuristic
        Ask yourself: **"Could the person searching for [Query] potentially find value in [Document Content]?"**
        * If YES (even "1%" chance): **KEEP**.
        * If NO (impossible): **REMOVE**.

        ### Output Format
        The output queries must be equal to the input! Return the result in JSON format with two distinct lists:

        {{
        "relevant_queries": ["list", "of", "queries", "to", "keep"],
        "filtered_queries": ["list", "of", "removed", "queries"]
        }}
        """

        structured_llm = self.llm.with_structured_output(DomainFilterResult)
        query_str = json.dumps(queries, indent=2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Document Summary:\n{doc_summary}\n\nQueries to evaluate:\n{query_str}")
        ])

        chain = prompt | structured_llm
        result = chain.invoke({"doc_summary": doc_summary, "query_str": query_str})
        print(f"   Kept: {len(result.relevant_queries)}, Filtered: {len(result.filtered_queries)}")
        return result

    # ----------------------------------------------------------------
    # Main Process
    # ----------------------------------------------------------------
    def process(self, html_content: str, file_uuid: str = None) -> FinalOutput:
        if not file_uuid: file_uuid = str(uuid.uuid4())

        title = self._extract_title(html_content)
        doc_summary = self._extract_summary(html_content)
        print(f"Processing: {title}")

        # 1. Get Profile
        profile = self.generate_profile(title, doc_summary)

        # 2. Filter 1: Determine applicable intents
        applicable = self.determine_applicable_intents(title, doc_summary)

        # Flatten keywords
        all_keywords = (
            profile.keyword_cluster.core[:2] +
            profile.keyword_cluster.lsi_synonyms[:2] +
            profile.keyword_cluster.keyphrases[:2]
        )

        print(f"   Keywords: {len(all_keywords)} extracted")
        print(f"   Personas: {[p.name for p in profile.target_personas]}")

        detailed_groups = []
        all_queries_flat = []

        # 3. Double loop (Keywords x Personas) with applicable intents
        for kw in all_keywords:
            variations = []
            print(f"   -> Processing Keyword: [{kw}]")

            for persona_obj in profile.target_personas:
                try:
                    # Generate queries only for applicable intents
                    intent_res = self.generate_queries_for_persona(
                        kw, persona_obj,
                        applicable_intents=applicable.intents
                    )

                    variations.append(PersonaVariation(
                        target_persona=persona_obj.name,
                        queries=intent_res
                    ))

                    flat_list = (
                        intent_res.navigational +
                        intent_res.informational +
                        intent_res.commercial +
                        intent_res.transactional
                    )
                    all_queries_flat.extend(flat_list)


                except Exception as e:
                    print(f"Error on {kw}/{persona_obj.name}: {e}")

            detailed_groups.append(KeywordGroup(
                focus_keyword=kw,
                variations=variations
            ))

        # 4. Deduplication
        print(f"   Raw queries before deduplication: {len(all_queries_flat)}")
        unique_queries = self.deduplicate_queries(all_queries_flat).unique_queries
        print(f"   Unique queries after deduplication: {len(unique_queries)}")

        # 5. Filter 2: Domain relevance filter
        filter_result = self.filter_by_domain(doc_summary, unique_queries)
        final_queries = filter_result.relevant_queries
        filtered_count = len(filter_result.filtered_queries)
        print(f"   Final queries after domain filter: {len(final_queries)} (filtered: {filtered_count})")

        # 6. Train / Test Split
        if len(final_queries) > 5:
            # 确保train至少60个，test至少20个
            if len(final_queries) >= 80:
                # 对于80个以上的查询，使用75%/25%的分割比例
                # 这样可以确保train至少60个，test至少20个
                train_size = max(60, int(len(final_queries) * 0.75))
                test_size = len(final_queries) - train_size
                train, test = train_test_split(final_queries, train_size=train_size, test_size=test_size, random_state=42)
            elif len(final_queries) >= 60:
                # 对于60-79个查询，优先保证train至少60个
                train = final_queries[:60]
                test = final_queries[60:]
            else:
                # 对于少于60个查询，使用50%/50%的分割比例
                train, test = train_test_split(final_queries, test_size=0.5, random_state=42)
        else:
            # 对于5个或更少的查询，全部作为train
            train, test = final_queries, []

        # 7. Build final output
        return FinalOutput(
            uuid=file_uuid,
            source_title=title,
            keywords=profile.keyword_cluster,
            personas=profile.target_personas,
            detailed_data=detailed_groups,
            dataset=DatasetSplit(train=train, test=test),
            stats={
                "total_unique": len(final_queries),
                "train_count": len(train),
                "test_count": len(test),
                "keywords_count": len(all_keywords),
                "personas_count": len(profile.target_personas),
                "applicable_intents": applicable.intents,
                "filtered_by_domain": filtered_count
            }
        )
