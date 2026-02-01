"""
Policy Engine for GEO Agent
将诊断结果 + 历史信号 → 工具选择的显式规则
替代纯 Prompt 依赖，提供更强的约束力

基于完整的失败分类体系设计策略映射
"""
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from geo_agent.core.telemetry import TelemetryStore, FailureCategory, ToolInvocationSpan


class PolicyDecision(Enum):
    """策略决策类型"""
    FORCE_TOOL = "force_tool"       # 强制使用某工具
    SUGGEST_TOOL = "suggest_tool"   # 建议使用某工具
    BLOCK_TOOL = "block_tool"       # 禁止使用某工具
    ESCALATE = "escalate"           # 升级策略（尝试更激进的方法）
    SKIP = "skip"                   # 跳过优化（无法修复）


@dataclass
class PolicyRule:
    """单条策略规则"""
    name: str
    priority: int  # 数字越小优先级越高
    decision: PolicyDecision
    target_tool: Optional[str]
    reason: str


@dataclass
class PolicyEvaluation:
    """策略评估结果"""
    forced_tool: Optional[str] = None
    suggested_tools: List[str] = None
    blocked_tools: List[str] = None
    applied_rules: List[PolicyRule] = None
    injection_prompt: str = ""  # 注入到 LLM 的策略提示
    should_skip: bool = False   # 是否应跳过优化
    skip_reason: str = ""
    
    def __post_init__(self):
        self.suggested_tools = self.suggested_tools or []
        self.blocked_tools = self.blocked_tools or []
        self.applied_rules = self.applied_rules or []


# 失败类型 -> 推荐工具的映射
FAILURE_TO_TOOL_MAP = {
    # Technical Issues - 部分可修复
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
    
    # Relevance Issues - 可能需要跳过
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
    策略引擎 - 基于诊断和历史做出决策
    
    设计原则：
    1. 硬性规则优先（如截断→必须重定位）
    2. 去重规则其次（避免重复失败）
    3. 升级规则最后（尝试不同策略）
    """
    
    # 工具分类
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
        评估当前状态，返回策略决策
        基于完整的失败分类体系
        """
        evaluation = PolicyEvaluation()
        
        # 收集所有适用的规则
        rules = []
        
        # ========== 0. 不可修复情况检测 ==========
        
        # 语义完全无关 + 高严重性 → 可能需要跳过
        if diagnosis_category == FailureCategory.SEMANTIC_IRRELEVANCE and severity == "critical":
            evaluation.should_skip = True
            evaluation.skip_reason = "Document is fundamentally irrelevant to the query - optimization may not help"
            # 仍然尝试一次
            rules.append(PolicyRule(
                name="LAST_RESORT_REALIGNMENT",
                priority=10,
                decision=PolicyDecision.SUGGEST_TOOL,
                target_tool="intent_realignment",
                reason="Attempting intent realignment as last resort for irrelevant content"
            ))
        
        # ========== 1. 基于诊断类型的工具映射 ==========
        
        if diagnosis_category in FAILURE_TO_TOOL_MAP:
            mapping = FAILURE_TO_TOOL_MAP[diagnosis_category]
            tool_list = mapping["tools"]
            priority_type = mapping["priority"]
            note = mapping["note"]
            
            # 选择第一个未被过度使用的工具
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
        
        # ========== 2. 截断特殊处理（覆盖其他规则） ==========
        
        if has_truncation_alert and hidden_content_summary:
            rules.append(PolicyRule(
                name="TRUNCATION_FORCE_RELOCATION",
                priority=0,  # 最高优先级
                decision=PolicyDecision.FORCE_TOOL,
                target_tool="content_relocation",
                reason=f"Hidden relevant content detected: {hidden_content_summary[:100]}..."
            ))
        
        # ========== 3. 去重规则 ==========
        
        # Rule 3.1: 同一工具连续失败 2 次 → 禁止
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
        
        # Rule 3.2: 某工具已尝试 3 次以上 → 禁止
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
        
        # ========== 4. 升级规则 ==========
        
        # Rule 4.1: 内容工具失败 + 信息缺失类问题 → 升级到策略工具
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
        
        # Rule 4.2: 结构工具失败 + 答案定位问题 → 尝试 BLUF
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
        
        # Rule 4.3: 多次截断警报 → 考虑移除噪音
        if self.telemetry.get_truncation_alerts_count() >= 2:
            rules.append(PolicyRule(
                name="REPEATED_TRUNCATION_ALERT",
                priority=2,
                decision=PolicyDecision.SUGGEST_TOOL,
                target_tool="noise_isolation",
                reason="Multiple truncation alerts - consider removing noise to fit more content"
            ))
        
        # ========== 应用规则 ==========
        
        # 按优先级排序
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
        
        # ========== 生成注入 Prompt ==========
        evaluation.injection_prompt = self._build_injection_prompt(evaluation)
        
        return evaluation
    
    def _build_injection_prompt(self, evaluation: PolicyEvaluation) -> str:
        """构建注入到 LLM 的策略提示"""
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
        
        # Skip 警告
        if evaluation.should_skip:
            lines.append(f"\n⚠️ **WARNING**: {evaluation.skip_reason}")
            lines.append("This optimization attempt may have limited effectiveness.")
        
        # 通用规则
        lines.append("\n**GENERAL RULES**:")
        lines.append("1. NEVER repeat the exact same tool + target_chunk combination that previously failed.")
        lines.append("2. If content injection failed twice, switch to persuasion or restructuring.")
        lines.append("3. Prioritize tools that address the ROOT CAUSE, not symptoms.")
        lines.append("4. For BURIED_ANSWER issues, always consider BLUF optimization first.")
        lines.append("5. For NOISE issues, use noise_isolation before structure_optimization.")
        
        return "\n".join(lines)
    
    def check_duplicate_invocation(self, tool_name: str, args_hash: str) -> Tuple[bool, str]:
        """
        检查是否为重复调用
        Returns: (is_duplicate, warning_message)
        """
        if self.telemetry.has_repeated_tool_args(tool_name, args_hash):
            return True, f"⚠️ Duplicate invocation detected: {tool_name} with same arguments was already tried."
        return False, ""
    
    def get_recommended_tools_for_category(self, category: FailureCategory) -> List[str]:
        """
        获取特定失败类型的推荐工具列表
        """
        if category in FAILURE_TO_TOOL_MAP:
            return FAILURE_TO_TOOL_MAP[category]["tools"]
        return []
    
    def is_category_fixable(self, category: FailureCategory) -> Tuple[bool, str]:
        """
        判断某个失败类型是否可以通过工具修复
        Returns: (is_fixable, reason)
        """
        # 完全无法修复的情况
        unfixable = {
            FailureCategory.SEMANTIC_IRRELEVANCE: "Document is fundamentally off-topic",
        }
        
        # 困难但可尝试的情况
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
