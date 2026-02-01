#!/usr/bin/env python3
"""
快速return_res实现，阶段间直接传递数据，不保存文件
"""

import json
import os
import logging
import time
import argparse
import asyncio
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


import numpy as np

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FastAttrConfig:
    """Fast属性评估器配置"""
    config_file: str
    use_fast_mode: bool = True
    max_concurrency: int = 8
    debug_mode: bool = False

@dataclass
class AttrEvaluationResult:
    """属性评估结果"""
    citations: List[int]
    answer_prompt: str
    highlight_set: List[Dict]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "citations": self.citations,
            "answer_prompt": self.answer_prompt,
            "highlight_set": self.highlight_set
        }

class FastAttrEvaluator:
    """快速属性评估器，阶段间直接传递数据"""
    
    def __init__(self, config: Optional[FastAttrConfig] = None, config_file: Optional[str] = None):
        """
        初始化快速属性评估器
        
        Args:
            config: 配置对象
            config_file: 配置文件路径（当config为None时使用）
        """
        if config is None:
            if config_file is None:
                # 使用默认配置文件
                base_dir = Path(__file__).resolve().parent
                config_file = str(base_dir / "configs/test/LFQA/full_pipeline.json")
            self.config = FastAttrConfig(config_file=config_file)
        else:
            # 确保config_file不为None
            if config.config_file is None:
                base_dir = Path(__file__).resolve().parent
                config.config_file = str(base_dir / "configs/test/LFQA/full_pipeline.json")
            self.config = config
        
        self.full_configs = None
        self.base_dir = Path(__file__).resolve().parent
        self._load_configs()
        # 导入所需的模块
        from .utils import (
            prompt_model, get_token_counter
        )
        from .subtask_specific_utils import (
            construct_prompts, get_subtask_funcs, get_subtask_prompt_structures, get_data
        )
        from .run_script import update_args
        
        self.construct_prompts = construct_prompts
        self.prompt_model = prompt_model
        self.get_subtask_funcs = get_subtask_funcs
        self.get_subtask_prompt_structures = get_subtask_prompt_structures
        self.get_token_counter = get_token_counter
        self.get_data = get_data
        self.update_args = update_args
    
    def _load_configs(self):
        """加载配置文件"""
        # 确保config_file是字符串
        if not isinstance(self.config.config_file, str):
            base_dir = Path(__file__).resolve().parent
            self.config.config_file = str(base_dir / "configs/test/LFQA/full_pipeline.json")
        
        config_path = Path(self.config.config_file)
        if not config_path.is_absolute():
            config_path = (self.base_dir / config_path).resolve()
        
        with open(config_path, 'r') as f:
            self.full_configs = json.load(f)
        
        # 验证配置
        if not any(elem['subtask'] == "content_selection" for elem in self.full_configs):
            raise ValueError("must provide content_selection configs")
        
        subtasks = set([elem['subtask'] for elem in self.full_configs])
        if subtasks not in [{"content_selection", "clustering"}, {"content_selection", "fusion_in_context"}]:
            raise ValueError('configs must be of the following subtasks: (1) "content_selection", "clustering" or (2) "content_selection", "fusion_in_context"')
        
        # 加载子任务配置
        self.subtask_configs = {}
        for elem in self.full_configs:
            subtask_name = elem['subtask']
            cfg_path = Path(elem['config_file'])
            if not cfg_path.is_absolute():
                cfg_path = (self.base_dir / cfg_path).resolve()
            
            with open(cfg_path, 'r') as f:
                subtask_cfg = json.load(f)
                self.subtask_configs[subtask_name] = subtask_cfg
                elem['config_file'] = str(cfg_path)
        
        # 检查所有子任务的split和setting是否一致
        all_splits = [cfg['split'] for cfg in self.subtask_configs.values()]
        all_settings = [cfg['setting'] for cfg in self.subtask_configs.values()]
        
        if len(set(all_splits)) != 1 or len(set(all_settings)) != 1:
            raise ValueError("all subtasks must have the same split (test/dev) and the same setting (MDS/LFQA)")
        
        self.split = all_splits[0]
        self.setting = all_settings[0]
    
    def _load_prompt_dict(self, setting: str) -> Dict:
        """加载提示字典"""
        prompt_file = self.base_dir / f"prompts/{setting}.json"
        with open(prompt_file, 'r') as f:
            return json.load(f)
    
    def _get_subtask_args(self, subtask_name: str, data: Dict) -> argparse.Namespace:
        """获取子任务的参数"""
        from argparse import Namespace
        
        # 获取子任务配置
        curr_configs = [elem for elem in self.full_configs if elem['subtask'] == subtask_name][0]
        
        # 创建基础参数
        args_dict = {
            'config_file': curr_configs['config_file'],
            'split': self.split,
            'setting': self.setting,
            'subtask': subtask_name,
            'indir_alignments': None,
            'outdir': str(self.base_dir / f"temp_{subtask_name}_{time.time()}"),
            'model_name': curr_configs.get('model_name', 'gpt-4.1-mini'),
            'n_demos': curr_configs.get('n_demos', 2),
            'num_retries': curr_configs.get('num_retries', 1),
            'temperature': curr_configs.get('temperature', 0.2),
            'debugging': self.config.debug_mode,
            'merge_cross_sents_highlights': False,
            'CoT': curr_configs.get('CoT', False),
            'cut_surplus': False,
            'prct_surplus': None,
            'always_with_question': False
        }
        
        # 更新配置文件中的参数
        args = Namespace(**args_dict)
        args = self.update_args(args)
        
        return args
    
    def _run_subtask(self, subtask_name: str, data: Dict, subtask_config: Dict) -> Dict:
        """运行子任务"""
        logger.info(f"\n=== 开始 {subtask_name} 阶段 ===")
        start_time = time.time()
        
        # 获取子任务参数
        args = self._get_subtask_args(subtask_name, data)
        
        # 准备数据
        prompt_dict = self._load_prompt_dict(self.setting)
        
        # 将单个数据实例包装成alignments_dict格式
        alignments_dict = [{
            "id": data["id"],
            "question": data["question"],
            "document": data["document"],
            "set_of_highlights_in_context": data.get("set_of_highlights_in_context", []),
            "response": ""
        }]
        
        # 获取子任务相关函数
        parse_response_func, convert_to_pipeline_style_func = self.get_subtask_funcs(args.subtask)
        
        # 获取子任务相关提示结构
        specific_prompt_details = self.get_subtask_prompt_structures(
            prompt_dict=prompt_dict, 
            setting=args.setting, 
            subtask=args.subtask, 
            CoT=args.CoT, 
            always_with_question=args.always_with_question
        )
        
        # 构建提示
        used_demos, prompts, additional_data = self.construct_prompts(
            prompt_dict=prompt_dict, 
            alignments_dict=alignments_dict, 
            n_demos=args.n_demos, 
            debugging=args.debugging, 
            merge_cross_sents_highlights=args.merge_cross_sents_highlights, 
            specific_prompt_details=specific_prompt_details,
            tkn_counter=self.get_token_counter(args.model_name),
            no_highlights=args.subtask in ["content_selection", "e2e_only_setting", "ALCE", "iterative_blue_print"],
            cut_surplus=args.cut_surplus,
            prct_surplus=args.prct_surplus
        )
        
        # 调用模型
        responses = self.prompt_model(
            prompts=prompts, 
            model_name=args.model_name, 
            parse_response_fn=parse_response_func, 
            num_retries=args.num_retries, 
            temperature=args.temperature
        )
        
        # 处理响应
        final_results = {key: dict() for key in responses.keys()}
        for instance_name, resp in responses.items():
            final_results[instance_name].update(additional_data[instance_name])
            final_results[instance_name].update(resp)
        
        # 转换为管道格式
        pipeline_format_results = None
        if convert_to_pipeline_style_func:
            try:
                # 根据不同的子任务，处理alignments_dict的不同格式要求
                if args.subtask == "clustering":
                    # 聚类任务的转换函数期望alignments_dict是列表格式
                    if isinstance(alignments_dict, dict):
                        # 如果是字典，转换回列表
                        alignments_dict_list = list(alignments_dict.values())
                        pipeline_format_results = convert_to_pipeline_style_func(final_results, alignments_dict_list)
                    else:
                        pipeline_format_results = convert_to_pipeline_style_func(final_results, alignments_dict)
                else:
                    # 其他任务期望alignments_dict是字典格式
                    if isinstance(alignments_dict, list):
                        # 创建一个字典，key是id，value是实例
                        alignments_dict_dict = {elem['id']: elem for elem in alignments_dict}
                        pipeline_format_results = convert_to_pipeline_style_func(final_results, alignments_dict_dict)
                    else:
                        pipeline_format_results = convert_to_pipeline_style_func(final_results, alignments_dict)
            except Exception as e:
                logger.error(f"转换为管道格式失败: {e}")
        
        end_time = time.time()
        logger.info(f"✅ {subtask_name} 阶段完成 (耗时: {end_time - start_time:.4f} 秒)")
        
        if pipeline_format_results and len(pipeline_format_results) > 0:
            return pipeline_format_results[0]
        else:
            # 如果转换失败，返回原始数据
            result = data.copy()
            result["response"] = ""
            return result
    
    def _run_content_selection(self, data: Dict, subtask_config: Dict) -> Dict:
        """运行Content Selection阶段"""
        return self._run_subtask("content_selection", data, subtask_config)
    
    def _run_clustering(self, data: Dict, subtask_config: Dict) -> Dict:
        """运行Clustering阶段"""
        return self._run_subtask("clustering", data, subtask_config)
    
    def _count_citations(self, data: Dict) -> List[int]:
        """计算引用计数"""
        if not data.get("document"):
            return []
        
        num_documents = len(data["document"])
        citations = [0] * num_documents
        
        # 统计每个文档的引用次数
        highlights = data.get("set_of_highlights_in_context", [])
        cited_indices = {int(h.get('documentFile', 0)) for h in highlights if h.get('documentFile') is not None}
        
        for idx in cited_indices:
            if 0 <= idx < num_documents:
                citations[idx] = 1
        
        return citations
    
    async def evaluate_async(self, id: str, query: str, rewritten_docs: List[str]) -> AttrEvaluationResult:
        """
        异步评估文档属性
        
        Args:
            id: 实例ID
            query: 查询文本
            rewritten_docs: 重写后的文档列表
            
        Returns:
            AttrEvaluationResult: 包含引用标记和回答文本的评估结果
        """
        logger.info(f"\n=== AttrEvaluator 快速模式开始运行 ===")
        logger.info(f"实例ID: {id}")
        logger.info(f"查询: {query[:100]}...")
        logger.info(f"文档数量: {len(rewritten_docs)}")
        
        # 准备输入数据
        input_data = {
            "id": id,
            "question": query,
            "document": [
                {"raw_text": doc, "url": "", "cleaned_text": doc}
                for doc in rewritten_docs
            ],
            "set_of_highlights_in_context": []
        }
        
        # 运行Content Selection阶段
        content_selection_result = await asyncio.get_event_loop().run_in_executor(
            None, 
            self._run_content_selection, 
            input_data, 
            self.subtask_configs["content_selection"]
        )
        
        final_result = content_selection_result
        
        # 运行Clustering阶段（如果配置了）
        if "clustering" in self.subtask_configs:
            clustering_result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._run_clustering, 
                content_selection_result, 
                self.subtask_configs["clustering"]
            )
            final_result = clustering_result
        
        # 计算引用计数
        citations = self._count_citations(final_result)
        
        # 准备最终结果
        result = AttrEvaluationResult(
            citations=citations,
            answer_prompt=final_result.get("response", ""),
            highlight_set=final_result.get("set_of_highlights_in_context", [])
        )
        
        logger.info(f"\n=== 生成最终答案阶段 ===")
        logger.info(f"生成的答案: {result.answer_prompt[:150]}...")
        logger.info(f"引用计数: {result.citations}")
        logger.info(f"高亮集合大小: {len(result.highlight_set)}")
        logger.info(f"高亮详情: {[(h.get('documentFile', 0)+1, h.get('highlight_text', '')[:50]+'...') for h in result.highlight_set[:3]]}{'...' if len(result.highlight_set) > 3 else ''}")
        logger.info("✅ 生成最终答案阶段完成")
        
        logger.info("\n=== AttrEvaluator 快速模式运行结束 ===")
        return result
    
    def evaluate(self, id: str, query: str, rewritten_docs: List[str]) -> Dict:
        """
        同步评估文档属性
        
        Args:
            id: 实例ID
            query: 查询文本
            rewritten_docs: 重写后的文档列表
            
        Returns:
            Dict: 包含引用标记和回答文本的字典
        """
        try:
            # 检查是否在事件循环内
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环内，使用现有循环运行异步函数
            result = loop.run_until_complete(self.evaluate_async(id, query, rewritten_docs))
            return result.to_dict()
        except RuntimeError:
            # 不在事件循环内，直接运行
            result = asyncio.run(self.evaluate_async(id, query, rewritten_docs))
            return result.to_dict()
    
    def return_res(self, id: str, query: str, rewritten_docs: List[str]) -> Dict:
        """
        快速版return_res接口（向后兼容）
        
        Args:
            id: 实例ID
            query: 查询文本
            rewritten_docs: 重写后的文档列表
            
        Returns:
            包含引用标记和回答文本的字典
        """
        # 保持向后兼容性，调用evaluate方法
        return self.evaluate(id, query, rewritten_docs)

def fast_return_res(id: str, query: str, rewritten_docs: List[str], config_file: Optional[str] = None, use_fast_mode: bool = True) -> Dict:
    """
    return_res接口，支持快速模式和原始模式切换
    
    Args:
        id: 实例ID
        query: 查询文本
        rewritten_docs: 重写后的文档列表
        config_file: 配置文件路径（可选）
        use_fast_mode: 是否使用快速模式，默认为True
        
    Returns:
        包含引用标记和回答文本的字典
    """
    # 使用默认配置文件
    if config_file is None:
        base_dir = Path(__file__).resolve().parent
        config_file = str(base_dir / "configs/test/LFQA/full_pipeline.json")
    
    if use_fast_mode:
        # 使用快速模式
        evaluator = FastAttrEvaluator(config_file=config_file)
        return evaluator.return_res(id, query, rewritten_docs)
    else:
        # 使用原始模式
        from .run_dataset import return_res, pre_init
        arg_list = ["--config-file", config_file]
        args = pre_init(arg_list)
        return return_res(id, query, rewritten_docs, args)

async def evaluate_async(
    id: str, 
    query: str, 
    rewritten_docs: List[str],
    config: Optional[FastAttrConfig] = None,
    config_file: Optional[str] = None
) -> AttrEvaluationResult:
    """
    异步评估文档属性的便捷函数
    
    Args:
        id: 实例ID
        query: 查询文本
        rewritten_docs: 重写后的文档列表
        config: 配置对象（可选）
        config_file: 配置文件路径（可选，当config为None时使用）
        
    Returns:
        AttrEvaluationResult: 包含引用标记和回答文本的评估结果
    """
    evaluator = FastAttrEvaluator(config=config, config_file=config_file)
    return await evaluator.evaluate_async(id, query, rewritten_docs)


def evaluate(
    id: str, 
    query: str, 
    rewritten_docs: List[str],
    config: Optional[FastAttrConfig] = None,
    config_file: Optional[str] = None
) -> Dict:
    """
    同步评估文档属性的便捷函数
    
    Args:
        id: 实例ID
        query: 查询文本
        rewritten_docs: 重写后的文档列表
        config: 配置对象（可选）
        config_file: 配置文件路径（可选，当config为None时使用）
        
    Returns:
        Dict: 包含引用标记和回答文本的字典
    """
    evaluator = FastAttrEvaluator(config=config, config_file=config_file)
    return evaluator.evaluate(id, query, rewritten_docs)

# 测试代码
if __name__ == "__main__":
    # 示例用法
    test_id = "test_001"
    test_query = "What is the capital of France?"
    test_docs = [
        "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023 in an area of more than 105 km² (41 sq mi), Paris is the fifth-most populated city in the European Union and the 30th most densely populated city in the world in 2022.",
        "France, officially the French Republic, is a country located primarily in Western Europe. It also includes overseas regions and territories in the Americas and the Atlantic, Pacific and Indian Oceans, giving it one of the largest discontiguous exclusive economic zones in the world."
    ]
    
    # 测试同步接口
    print("=== 测试同步接口 ===")
    result = evaluate(test_id, test_query, test_docs)
    print(f"引用计数: {result['citations']}")
    print(f"回答: {result['answer_prompt'][:100]}...")
    print(f"高亮数量: {len(result['highlight_set'])}")
    
    # 测试异步接口
    print("\n=== 测试异步接口 ===")
    async def test_async():
        result = await evaluate_async(test_id, test_query, test_docs)
        print(f"引用计数: {result.citations}")
        print(f"回答: {result.answer_prompt[:100]}...")
        print(f"高亮数量: {len(result.highlight_set)}")
    
    asyncio.run(test_async())
