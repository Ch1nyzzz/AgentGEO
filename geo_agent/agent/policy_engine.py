"""
Policy Engine for GEO Agent
Converts diagnosis results + historical signals → explicit rules for tool selection
Replaces pure Prompt dependency with stronger constraints

Strategy mapping designed based on complete failure classification system
"""
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from geo_agent.core.telemetry import TelemetryStore, FailureCategory, ToolInvocationSpan


class PolicyDecision(Enum):
    """Policy decision type"""
    FORCE_TOOL = "force_tool"       # Force use of a specific tool
    SUGGEST_TOOL = "suggest_tool"   # Suggest use of a specific tool
    BLOCK_TOOL = "block_tool"       # Block use of a specific tool
    ESCALATE = "escalate"           # Escalate strategy (try more aggressive approach)
    SKIP = "skip"                   # Skip optimization (cannot be fixed)


@dataclass
class PolicyRule:
    """Single policy rule"""
    name: str
    priority: int  # 数字越小优先级越高
    decision: PolicyDecision
    target_tool: Optional[str]
    reason: str


@dataclass
class PolicyEvaluation:
    """Policy evaluation result"""
    forced_tool: Optional[str] = None
    suggested_tools: List[str] = None
    blocked_tools: List[str] = None
    applied_rules: List[PolicyRule] = None
    injection_prompt: str = ""  # Policy prompt injected into LLM
    should_skip: bool = False   # Whether optimization should be skipped
    skip_reason: str = ""
    
    def __post_init__(self):
        self.suggested_tools = self.suggested_tools or []
        self.blocked_tools = self.blocked_tools or []
        self.applied_rules = self.applied_rules or []


# Failure type -> Recommended tool mapping
FAILURE_TO_TOOL_MAP = {
    # Technical Issues - Partially fixable
    FailureCategory.PARSING_FAILURE: {
        "tools": ["noise_isolation", "structure_optimization"],
        "priority": "suggest",
        "note": "May require manual intervention if parsing is fundamentally broken"
    },
    FailureCategory.CONTENT_TRUNCATED: {
        "tools": ["content_relocation"],
        "priority": "force",
        "note": "Must surface hidden content"
    },
    FailureCategory.DATA_INTEGRITY: {
        "tools": ["noise_isolation"],
        "priority": "suggest",
        "note": "Try to clean up extraction artifacts"
    },
    
    # Noise Issues
    FailureCategory.WEB_NOISE: {
        "tools": ["noise_isolation", "bluf_optimization"],
        "priority": "force",
        "note": "Remove boilerplate and navigation elements"
    },
    FailureCategory.LOW_SIGNAL_RATIO: {
        "tools": ["noise_isolation", "structure_optimization"],
        "priority": "suggest",
        "note": "Improve signal by removing noise and restructuring"
    },
    
    # Density Issues
    FailureCategory.LOW_INFO_DENSITY: {
        "tools": ["entity_injection", "data_serialization"],
        "priority": "suggest",
        "note": "Add more specific facts and data"
    },
    FailureCategory.MISSING_INFO: {
        "tools": ["entity_injection", "data_serialization"],
        "priority": "force",
        "note": "Inject missing information"
    },
    
    # Structure Issues
    FailureCategory.STRUCTURAL_WEAKNESS: {
        "tools": ["structure_optimization", "bluf_optimization"],
        "priority": "force",
        "note": "Improve document structure and segmentation"
    },
    
    # Relevance Issues - May need to skip
    FailureCategory.SEMANTIC_IRRELEVANCE: {
        "tools": ["intent_realignment"],
        "priority": "suggest",
        "note": "Try to realign content with query intent, but may be unfixable"
    },
    FailureCategory.ATTRIBUTE_MISMATCH: {
        "tools": ["intent_realignment", "entity_injection"],
        "priority": "suggest",
        "note": "Realign to correct attributes or inject missing ones"
    },
    
    # Answer Positioning
    FailureCategory.BURIED_ANSWER: {
        "tools": ["bluf_optimization", "content_relocation", "structure_optimization"],
        "priority": "force",
        "note": "Surface the buried answer to the top"
    },
    
    # Quality Issues
    FailureCategory.NON_FACTUAL_CONTENT: {
        "tools": ["entity_injection", "data_serialization"],
        "priority": "suggest",
        "note": "Add factual content to balance opinions"
    },
    FailureCategory.TRUST_CREDIBILITY: {
        "tools": ["persuasive_rewriting", "entity_injection"],
        "priority": "suggest",
        "note": "Add authoritative language and citations"
    },
    
    # Temporal Issues
    FailureCategory.OUTDATED_CONTENT: {
        "tools": ["entity_injection"],
        "priority": "suggest",
        "note": "Inject updated information if available"
    },
}


class PolicyEngine:
    """
    Policy Engine - Makes decisions based on diagnosis and history

    Design principles:
    1. Hard rules first (e.g., truncation → must relocate)
    2. Deduplication rules second (avoid repeated failures)
    3. Escalation rules last (try different strategies)
    """

    # Tool categories
    RESTRUCTURE_TOOLS = ["content_relocation", "structure_optimization", "noise_isolation"]
    CONTENT_TOOLS = ["entity_injection", "data_serialization"]
    STRATEGY_TOOLS = ["persuasive_rewriting", "historical_redteam", "intent_realignment"]
    
    def __init__(self, telemetry: TelemetryStore):
        self.telemetry = telemetry
    
    def evaluate(
        self,
        diagnosis_category: FailureCategory,
        diagnosis_explanation: str,
        has_truncation_alert: bool = False,
        hidden_content_summary: str = "",
        severity: str = "medium"
    ) -> PolicyEvaluation:
        """
        Evaluate current state and return policy decision
        Based on complete failure classification system
        """
        evaluation = PolicyEvaluation()

        # Collect all applicable rules
        rules = []

        # ========== 0. Unfixable Situation Detection ==========

        # Completely semantically irrelevant + high severity → may need to skip
        if diagnosis_category == FailureCategory.SEMANTIC_IRRELEVANCE and severity == "critical":
            evaluation.should_skip = True
            evaluation.skip_reason = "Document is fundamentally irrelevant to the query - optimization may not help"
            # Still try once
            rules.append(PolicyRule(
                name="LAST_RESORT_REALIGNMENT",
                priority=10,
                decision=PolicyDecision.SUGGEST_TOOL,
                target_tool="intent_realignment",
                reason="Attempting intent realignment as last resort for irrelevant content"
            ))
        
        # ========== 1. Tool Mapping Based on Diagnosis Type ==========
        
        if diagnosis_category in FAILURE_TO_TOOL_MAP:
            mapping = FAILURE_TO_TOOL_MAP[diagnosis_category]
            tool_list = mapping["tools"]
            priority_type = mapping["priority"]
            note = mapping["note"]
            
            # Select the first tool that hasn't been overused
            selected_tool = None
            for tool in tool_list:
                if self.telemetry.get_tool_usage_count(tool) < 3:
                    selected_tool = tool
                    break
            
            if selected_tool:
                if priority_type == "force":
                    rules.append(PolicyRule(
                        name=f"DIAGNOSIS_FORCE_{diagnosis_category.value.upper()}",
                        priority=1,
                        decision=PolicyDecision.FORCE_TOOL,
                        target_tool=selected_tool,
                        reason=f"{note}. Diagnosis: {diagnosis_category.value}"
                    ))
                else:
                    rules.append(PolicyRule(
                        name=f"DIAGNOSIS_SUGGEST_{diagnosis_category.value.upper()}",
                        priority=2,
                        decision=PolicyDecision.SUGGEST_TOOL,
                        target_tool=selected_tool,
                        reason=f"{note}. Diagnosis: {diagnosis_category.value}"
                    ))
        
        # ========== 2. Special Truncation Handling (Overrides Other Rules) ==========
        
        if has_truncation_alert and hidden_content_summary:
            rules.append(PolicyRule(
                name="TRUNCATION_FORCE_RELOCATION",
                priority=0,  # Highest priority
                decision=PolicyDecision.FORCE_TOOL,
                target_tool="content_relocation",
                reason=f"Hidden relevant content detected: {hidden_content_summary[:100]}..."
            ))
        
        # ========== 3. Deduplication Rules ==========

        # Rule 3.1: Same tool failed consecutively 2 times → block
        recent_tools = self.telemetry.get_recent_tools(n=2)
        if len(recent_tools) >= 2 and recent_tools[-1] == recent_tools[-2]:
            failed_tool = recent_tools[-1]
            rules.append(PolicyRule(
                name="CONSECUTIVE_FAILURE_BLOCK",
                priority=3,
                decision=PolicyDecision.BLOCK_TOOL,
                target_tool=failed_tool,
                reason=f"Tool '{failed_tool}' failed consecutively, blocking to force strategy change"
            ))
        
        # Rule 3.2: A tool has been tried 3+ times → block
        all_tools = self.RESTRUCTURE_TOOLS + self.CONTENT_TOOLS + self.STRATEGY_TOOLS
        for tool_name in all_tools:
            if self.telemetry.get_tool_usage_count(tool_name) >= 3:
                rules.append(PolicyRule(
                    name="OVERUSED_TOOL_BLOCK",
                    priority=4,
                    decision=PolicyDecision.BLOCK_TOOL,
                    target_tool=tool_name,
                    reason=f"Tool '{tool_name}' has been tried {self.telemetry.get_tool_usage_count(tool_name)} times without success"
                ))
        
        # ========== 4. Escalation Rules ==========

        # Rule 4.1: Content tools failed + info-missing issue → escalate to strategy tools
        content_attempts = sum(self.telemetry.get_tool_usage_count(t) for t in self.CONTENT_TOOLS)
        info_related = diagnosis_category in [
            FailureCategory.MISSING_INFO, 
            FailureCategory.LOW_INFO_DENSITY,
            FailureCategory.NON_FACTUAL_CONTENT
        ]
        if content_attempts >= 2 and info_related:
            strategy_attempts = sum(self.telemetry.get_tool_usage_count(t) for t in self.STRATEGY_TOOLS)
            if strategy_attempts == 0:
                rules.append(PolicyRule(
                    name="ESCALATE_TO_STRATEGY",
                    priority=5,
                    decision=PolicyDecision.SUGGEST_TOOL,
                    target_tool="persuasive_rewriting",
                    reason="Content injection tools exhausted, escalating to persuasive_rewriting strategy"
                ))
        
        # Rule 4.2: Structure tools failed + answer positioning issue → try BLUF
        restructure_attempts = sum(self.telemetry.get_tool_usage_count(t) for t in self.RESTRUCTURE_TOOLS)
        if restructure_attempts >= 2 and diagnosis_category == FailureCategory.BURIED_ANSWER:
            if self.telemetry.get_tool_usage_count("bluf_optimization") == 0:
                rules.append(PolicyRule(
                    name="ESCALATE_TO_BLUF",
                    priority=5,
                    decision=PolicyDecision.FORCE_TOOL,
                    target_tool="bluf_optimization",
                    reason="Restructure tools failed for buried answer, trying BLUF optimization"
                ))
        
        # Rule 4.3: Multiple truncation alerts → consider removing noise
        if self.telemetry.get_truncation_alerts_count() >= 2:
            rules.append(PolicyRule(
                name="REPEATED_TRUNCATION_ALERT",
                priority=2,
                decision=PolicyDecision.SUGGEST_TOOL,
                target_tool="noise_isolation",
                reason="Multiple truncation alerts - consider removing noise to fit more content"
            ))
        
        # ========== Apply Rules ==========

        # Sort by priority
        rules.sort(key=lambda r: r.priority)
        
        for rule in rules:
            evaluation.applied_rules.append(rule)
            
            if rule.decision == PolicyDecision.FORCE_TOOL:
                evaluation.forced_tool = rule.target_tool
            elif rule.decision == PolicyDecision.SUGGEST_TOOL:
                if rule.target_tool not in evaluation.suggested_tools:
                    evaluation.suggested_tools.append(rule.target_tool)
            elif rule.decision == PolicyDecision.BLOCK_TOOL:
                if rule.target_tool not in evaluation.blocked_tools:
                    evaluation.blocked_tools.append(rule.target_tool)
        
        # ========== Generate Injection Prompt ==========
        evaluation.injection_prompt = self._build_injection_prompt(evaluation)
        
        return evaluation
    
    def _build_injection_prompt(self, evaluation: PolicyEvaluation) -> str:
        """Build policy prompt to inject into LLM"""
        lines = ["### 🎯 OPTIMIZATION POLICY (SYSTEM ENFORCED)"]
        
        if evaluation.forced_tool:
            lines.append(f"**MANDATORY**: You MUST use `{evaluation.forced_tool}`. This is non-negotiable.")
            lines.append(f"Reason: {evaluation.applied_rules[0].reason}")
            return "\n".join(lines)
        
        if evaluation.blocked_tools:
            blocked_str = ", ".join(f"`{t}`" for t in evaluation.blocked_tools)
            lines.append(f"**BLOCKED TOOLS** (DO NOT USE): {blocked_str}")
            for rule in evaluation.applied_rules:
                if rule.decision == PolicyDecision.BLOCK_TOOL:
                    lines.append(f"  - {rule.target_tool}: {rule.reason}")
        
        if evaluation.suggested_tools:
            suggested_str = ", ".join(f"`{t}`" for t in evaluation.suggested_tools)
            lines.append(f"**RECOMMENDED TOOLS** (Prefer these): {suggested_str}")
        
        # Skip warning
        if evaluation.should_skip:
            lines.append(f"\n⚠️ **WARNING**: {evaluation.skip_reason}")
            lines.append("This optimization attempt may have limited effectiveness.")
        
        # General rules
        lines.append("\n**GENERAL RULES**:")
        lines.append("1. NEVER repeat the exact same tool + target_chunk combination that previously failed.")
        lines.append("2. If content injection failed twice, switch to persuasion or restructuring.")
        lines.append("3. Prioritize tools that address the ROOT CAUSE, not symptoms.")
        lines.append("4. For BURIED_ANSWER issues, always consider BLUF optimization first.")
        lines.append("5. For NOISE issues, use noise_isolation before structure_optimization.")
        
        return "\n".join(lines)
    
    def check_duplicate_invocation(self, tool_name: str, args_hash: str) -> Tuple[bool, str]:
        """
        Check if this is a duplicate invocation
        Returns: (is_duplicate, warning_message)
        """
        if self.telemetry.has_repeated_tool_args(tool_name, args_hash):
            return True, f"⚠️ Duplicate invocation detected: {tool_name} with same arguments was already tried."
        return False, ""
    
    def get_recommended_tools_for_category(self, category: FailureCategory) -> List[str]:
        """
        Get recommended tool list for a specific failure type
        """
        if category in FAILURE_TO_TOOL_MAP:
            return FAILURE_TO_TOOL_MAP[category]["tools"]
        return []
    
    def is_category_fixable(self, category: FailureCategory) -> Tuple[bool, str]:
        """
        Determine if a failure type can be fixed with tools
        Returns: (is_fixable, reason)
        """
        # Completely unfixable cases
        unfixable = {
            FailureCategory.SEMANTIC_IRRELEVANCE: "Document is fundamentally off-topic",
        }

        # Difficult but worth trying cases
        difficult = {
            FailureCategory.PARSING_FAILURE: "Parsing issues may require manual intervention",
            FailureCategory.OUTDATED_CONTENT: "Cannot automatically update temporal information",
            FailureCategory.DATA_INTEGRITY: "Data integrity issues may be unrecoverable",
        }
        
        if category in unfixable:
            return False, unfixable[category]
        if category in difficult:
            return True, f"Difficult: {difficult[category]}"
        return True, "Fixable with appropriate tools"
