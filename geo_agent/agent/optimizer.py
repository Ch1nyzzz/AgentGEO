import json
import time
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from geo_agent.tools import registry
from geo_agent.core.models import WebPage, AnalysisResult, OptimizationState
from geo_agent.core.memory import OptimizationMemory, ModificationRecord, parse_tool_output
from geo_agent.core.telemetry import (
    TelemetryStore, IterationMetrics, ToolInvocationSpan, 
    FailureCategory, ToolOutcome, compute_args_hash
)
from geo_agent.config import get_llm_from_config, get_generator_from_config, load_config
from geo_agent.search_engine import SearchManager
from geo_agent.agent.failure_analysis import analyze_failure, regenerate_tool_args, DiagnosisResult
from geo_agent.agent.content_auditor import audit_content_truncation
from geo_agent.agent.policy_engine import PolicyEngine

from geo_agent.utils.structural_parser import StructuralHtmlParser

class GEOAgent:
    def __init__(self, config_path='geo_agent/config.yaml'):
        self.config = load_config(config_path)
        self.llm = get_llm_from_config(config_path)
        self.search_manager = SearchManager(config_path)        
        self.generator = get_generator_from_config(config_path)
        self.tools_map = registry.tools
        
        # Define Pydantic Parsers
        self.analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

        # Get max_snippet_length setting
        gen_config = self.config.get("generator", {})
        self.max_snippet_length = gen_config.get("max_snippet_length", 10000)

        # Cache full content
        self._content_cache = {}


    def extract_core_idea(self, content: str) -> str:
        """
        Extract the core theme of the document as an anchor for subsequent optimization to prevent theme drift.
        """
        prompt = ChatPromptTemplate.from_template("""
        Analyze this document and extract its CORE IDEA in 2-3 sentences.

        Focus on:
        - What is the PRIMARY topic/subject of this document?
        - What is the main purpose or theme?
        - What key concepts must be preserved?

        Document:
        {content}

        CORE IDEA (2-3 sentences):
        """)

        # Use only the first 2000 chars to extract core idea
        response = self.llm.invoke(prompt.format(content=content))
        return response.content.strip()




    def _initialize_optimization(self, webpage: WebPage):
        """Initializes the optimization process (parser, core idea, memory, logs)."""
        output_dir = Path("geo_agent/outputs/optimization_logs")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = webpage.url.replace('https://', '').replace('http://', '').replace('/', '_')[:50]
        output_file = output_dir / f"{safe_filename}.json"

        # 1. Structural Parsing
        struct_parser = StructuralHtmlParser(min_length=50)
        if not webpage.raw_html:
            raise ValueError("WebPage.raw_html is required for optimization.")
        
        current_structure = struct_parser.parse(webpage.raw_html)
        doc_text = current_structure.get_clean_text()

        # 2. Extract Core Idea
        core_idea = self.extract_core_idea(doc_text)
        print(f"📌 Core Idea: {core_idea}")
        print("--------------------------------")

        memory = OptimizationMemory(core_idea=core_idea)

        # 3. Setup Logging
        log_data = {
            "url": webpage.url,
            "core_idea": core_idea,
            "initial_content": doc_text,
            "optimizations": [],
            "memory_history": []
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        return current_structure, struct_parser, memory, log_data, output_file, core_idea

    def _prepare_tool_arguments(self, analysis, current_structure, core_idea, memory):
        """Prepares the arguments for the selected tool."""
        tool_args = analysis.tool_arguments.copy()
        
        target_idx = analysis.target_chunk_index if analysis.target_chunk_index is not None else 0
        print(f"📦 Target chunk: {target_idx}")
        
        # Use structure manager to get context
        chunk_args = current_structure.get_chunk_tool_args(target_idx)
        tool_args.update(chunk_args)

        tool_args['core_idea'] = core_idea
        tool_args['previous_modifications'] = memory.get_preservation_rules()
        
        return tool_args, target_idx

    def optimize_page(self, webpage: WebPage, queries: List[str], max_retries: int = 3):
        print(f"Starting GEO for {webpage.url}")
        print("--------------------------------")

        current_structure, struct_parser, memory, log_data, output_file, core_idea = self._initialize_optimization(webpage)
        
        # Initialize Telemetry Store
        telemetry = TelemetryStore(url=webpage.url, core_idea=core_idea)
        policy_engine = PolicyEngine(telemetry)

        for idx, query in enumerate(queries):
            print(f"\n--- Processing Query: {query} ---")
            state = OptimizationState(query=query, current_content=current_structure.get_clean_text(), iteration=0)

            competitor_contents = self.search_manager.search_and_retrieve(query)

            success = False
            final_answer = ""

            for i in range(max_retries):
                print(f"Iteration {i+1}/{max_retries}")
                
                current_raw_html = current_structure.export_html()
                current_structure = struct_parser.parse(current_raw_html) 
                current_clean_text = current_structure.get_clean_text()
                
                # Check Citation
                temp_page = webpage.model_copy(update={"cleaned_content": current_clean_text})
                citation_result = self.generator.generate_and_check(query, temp_page, competitor_contents)
                
                if citation_result.is_cited:
                    print(f"✅ Success! Document cited for query: {query}")
                    success = True
                    final_answer = citation_result.generated_answer
                    break
                else:
                    print(f"❌ Not cited. Analyzing...")

                # Chunking
                current_structure.calculate_chunks(max_chunk_length=2000, max_snippet_length=self.max_snippet_length)

                # Analyze & Select Tool
                indexed_target_content = current_structure.format_indexed_content()
                full_doc_length = len(current_clean_text)
                visible_chunk_length = len(indexed_target_content)
                
                # --- Audit Truncation ---
                truncation_summary = None
                has_truncation_alert = False
                try:
                    audit_res = audit_content_truncation(
                        self.llm,
                        query,
                        full_text=current_clean_text,
                        visible_chunks_text=indexed_target_content
                    )
                    if audit_res.has_hidden_relevant_content:
                        print(f"⚠️ Truncation Alert: {audit_res.summary_of_hidden_info}")
                        truncation_summary = audit_res.summary_of_hidden_info
                        has_truncation_alert = True
                except Exception as e:
                    print(f"Audit warning: {e}")

                cited_doc = self.generator.get_one_cited_content(
                    citation_result.generated_answer, competitor_contents
                )
                
                # --- Policy Evaluation (Pre-Analysis) ---
                # 先用 truncation 信息做初步诊断分类
                pre_diagnosis_category = FailureCategory.CONTENT_TRUNCATED if has_truncation_alert else FailureCategory.UNKNOWN
                policy_eval = policy_engine.evaluate(
                    diagnosis_category=pre_diagnosis_category,
                    diagnosis_explanation="",
                    has_truncation_alert=has_truncation_alert,
                    hidden_content_summary=truncation_summary or ""
                )
                
                # --- Failure Analysis with Policy Injection ---
                analysis, diagnosis = analyze_failure(
                    self.llm,
                    query, indexed_target_content,
                    cited_doc, 
                    memory=memory,
                    truncation_audit_summary=truncation_summary,
                    policy_injection=policy_eval.injection_prompt
                )
                
                # 打印诊断信息
                diagnosis_category = diagnosis.to_category() if diagnosis else FailureCategory.UNKNOWN
                severity = getattr(diagnosis, 'severity', 'medium') if diagnosis else 'medium'
                print(f"🧐 Diagnosis: {diagnosis_category.value} (Severity: {severity})")
                print(f"   Explanation: {diagnosis.explanation[:100]}..." if diagnosis else "")
                print(f"Reason: {analysis.reasoning}")
                print(f"Tool Selected: {analysis.selected_tool_name}")
                
                # --- Re-evaluate Policy with actual diagnosis ---
                if diagnosis:
                    policy_eval = policy_engine.evaluate(
                        diagnosis_category=diagnosis_category,
                        diagnosis_explanation=diagnosis.explanation,
                        has_truncation_alert=has_truncation_alert,
                        hidden_content_summary=truncation_summary or "",
                        severity=severity
                    )
                    
                    # Check for skip recommendation
                    if policy_eval.should_skip:
                        print(f"⚠️ Policy Warning: {policy_eval.skip_reason}")
                    
                    # Check for forced tool override
                    if policy_eval.forced_tool and policy_eval.forced_tool != analysis.selected_tool_name:
                        print(f"🎯 Policy Override: {analysis.selected_tool_name} -> {policy_eval.forced_tool}")
                        original_tool = analysis.selected_tool_name

                        # 重新生成适配新工具的参数（修复参数不匹配问题）
                        try:
                            history_context = memory.get_history_summary() if memory and memory.modifications else ""
                            analysis = regenerate_tool_args(
                                llm=self.llm,
                                forced_tool=policy_eval.forced_tool,
                                diagnosis=diagnosis,
                                query=query,
                                target_content_indexed=indexed_target_content,
                                history_context=history_context,
                            )
                            print(f"✅ Regenerated args for {policy_eval.forced_tool}")
                        except Exception as e:
                            print(f"❌ Failed to regenerate args for {policy_eval.forced_tool}: {e}")
                            # 回退到原始工具
                            analysis.selected_tool_name = original_tool
                            print(f"⚠️ Falling back to original tool: {original_tool}")

                # Execute Tool
                tool = registry.get_tool(analysis.selected_tool_name)
                if not tool:
                    print(f"Error: Tool {analysis.selected_tool_name} not found.")
                    break

                tool_start_time = time.time()
                tool_outcome = ToolOutcome.FAILED
                error_msg = None
                
                try:
                    tool_args, target_idx = self._prepare_tool_arguments(analysis, current_structure, core_idea, memory)
                    
                    # Add extra args for content_relocation tool
                    if analysis.selected_tool_name == "content_relocation":
                        # 仅当 truncation_summary 有实际内容时才覆盖，否则保留 LLM 可能生成的值
                        if truncation_summary:
                            tool_args["hidden_content_summary"] = truncation_summary
                        elif "hidden_content_summary" not in tool_args:
                            tool_args["hidden_content_summary"] = ""
                        tool_args["query"] = query

                    # Add extra args for intent_realignment tool
                    if analysis.selected_tool_name == "intent_realignment":
                        tool_args["user_query"] = query

                    # Add extra args for historical_redteam tool
                    if analysis.selected_tool_name == "historical_redteam":
                        tool_args["target_query"] = query

                    # --- Duplicate Check ---
                    args_hash = compute_args_hash(tool_args)
                    is_dup, dup_msg = policy_engine.check_duplicate_invocation(analysis.selected_tool_name, args_hash)
                    if is_dup:
                        print(dup_msg)
                        # Still proceed but log it
                    
                    raw_output = tool.run(tool_args)
                    modified_chunk_html, key_changes = parse_tool_output(raw_output)

                    if current_structure.replace_chunk_by_index(target_idx, modified_chunk_html, highlight=True):
                         print("DOM updated successfully.")
                         tool_outcome = ToolOutcome.SUCCESS
                         
                         # Save for verification
                         debug_name = f"debug_iter{i}_{query.replace(' ', '_')[:20]}.html"
                         debug_path = output_file.parent / debug_name
                         with open(debug_path, 'w', encoding='utf-8') as f:
                             f.write(current_structure.export_html())
                         print(f"📄 Saved HTML for manual verification: {debug_path}")
                    else:
                         print("Failed to update DOM.")
                         tool_outcome = ToolOutcome.PARTIAL

                    memory.add_modification(ModificationRecord(
                        query=query,
                        tool_name=analysis.selected_tool_name,
                        reasoning=analysis.reasoning,
                        key_changes=key_changes
                    ))
                    print(f"📝 Key Changes: {key_changes}")
                    state.history.append(analysis.selected_tool_name)

                except Exception as e:
                    print(f"Tool execution failed: {e}")
                    error_msg = str(e)
                    import traceback
                    traceback.print_exc()
                
                tool_duration = (time.time() - tool_start_time) * 1000
                
                # --- Record Telemetry ---
                tool_span = ToolInvocationSpan(
                    tool_name=analysis.selected_tool_name,
                    target_chunk_index=target_idx if 'target_idx' in dir() else 0,
                    arguments_hash=args_hash if 'args_hash' in dir() else "",
                    outcome=tool_outcome,
                    reasoning=analysis.reasoning,
                    duration_ms=tool_duration,
                    error_message=error_msg
                )
                
                iteration_metrics = IterationMetrics(
                    iteration_index=i,
                    query=query,
                    full_doc_length=full_doc_length,
                    visible_chunk_length=visible_chunk_length,
                    truncation_ratio=1 - (visible_chunk_length / full_doc_length) if full_doc_length > 0 else 0,
                    chunk_count=len(current_structure._chunks) if hasattr(current_structure, '_chunks') else 0,
                    diagnosis_category=diagnosis.to_category() if diagnosis else FailureCategory.UNKNOWN,
                    diagnosis_explanation=diagnosis.explanation if diagnosis else "",
                    has_hidden_relevant_content=has_truncation_alert,
                    hidden_content_summary=truncation_summary or "",
                    tool_span=tool_span,
                    was_cited=False
                )
                telemetry.record_iteration(iteration_metrics)

            # End Retry Loop - Update WebPage & Logs
            webpage.raw_html = current_structure.export_html()
            webpage.cleaned_content = current_structure.get_clean_text()

            log_data["optimizations"].append({
                "query": query,
                "success": success,
                "final_answer": final_answer,
                "modified_content_text": webpage.cleaned_content,
                "tools_used": state.history
            })
            log_data["memory_history"] = memory.to_dict()["modifications"]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"Saved progress to {output_file}")
        
        # Export Telemetry
        telemetry_path = output_file.parent / f"{output_file.stem}_telemetry.json"
        telemetry.export_json(str(telemetry_path))
        print(f"📊 Telemetry exported to {telemetry_path}")

        print("Optimization complete. HTML updated directly via StructuralParser.")
        return webpage
    
    def evaluate_page(self, webpage: WebPage, queries: List[str]):
        """
        Evaluate the optimized page to see if it is cited.
        """
        results = {}
        for query in tqdm(queries, desc="Evaluating Queries"): 
            competitor_contents = self.search_manager.search_and_retrieve(query)
            citation_result = self.generator.generate_and_check(query, webpage, competitor_contents)
            results[query] = citation_result.is_cited
        success_count = sum(results.values())
        ratio = success_count / len(queries) if queries else 0
        results['ratio'] = ratio
        return results
    
    
