import os
import argparse
import json
from .utils import *
from .run_script import main as main_func
from .run_iterative_sentence_generation import main as iterative_sent_gen_main
import logging
from copy import deepcopy
from pathlib import Path
import time
from datasets import load_dataset
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

def run_subtask(full_configs, subtask_name, curr_outdir, original_args_dict, indir_alignments=None, choose_idx=None, attack_mode=None, data_num=None):
    """
    full_configs: full pipeline configs
    subtask_name: curr subtask name (either one of "content_selection", "clustering", "iterative_sentence_generation", or "fusion_in_context")
    curr_outdir: curr subtask's outdir
    original_args_dict: full otiginal args
    indir_alignments: path to previous subtask's alignments (for content_selection this should be None or a pre-defined alignment path)
    """
    curr_configs = [elem for elem in full_configs if elem['subtask']==subtask_name][0]

    curr_configs.update({"outdir":curr_outdir,
                         "indir_alignments":indir_alignments
                         })
    if subtask_name=="content_selection":
        curr_configs.update({"choose_idx": choose_idx,
                             "data_num":data_num,
                             "attack_mode": attack_mode
                             })

    func_args = deepcopy(original_args_dict) # initialize args that didn't appear in the subask's configs file to default values
    func_args.update(curr_configs)
    if subtask_name!="iterative_sentence_generation":
        return main_func(argparse.Namespace(**func_args))
    else:
        return iterative_sent_gen_main(argparse.Namespace(**func_args))

def compute_stability(pipes):
    """
    计算多次运行(pipes)之间，每个样本引用行为的稳定性。

    Args:
        pipes (list): 包含多次运行结果的嵌套列表。

    Returns:
        dict: 一个字典，key是样本ID，value是该样本的引用稳定性得分 (0到1之间)。
              得分1表示在所有运行中，对每个文档的引用行为完全一致。
    """
    stability_scores = {}
    unstable_examples = {}
    unstable_count=0
    if not pipes or not pipes[0]:
        return {}
        
    num_examples = len(pipes[0])
    num_pipes = len(pipes)

    # 遍历所有样本
    for i in range(num_examples):
        example_id = pipes[0][i].get('id')
        
        # 收集该样本在每次运行中的引用标记列表
        # 注意：这里修正了您草稿中循环变量重复的问题
        citation_history = [count_citations(pipes[j][i]) for j in range(num_pipes)]
        sum_citation = [sum(col) for col in zip(*citation_history)]
        for idx in range(len(sum_citation)):
            if sum_citation[idx]==0 or sum_citation[idx]==num_pipes:
                continue
            else:
                unstable_examples.update({"example_id": example_id, "doc_idx": idx, "citation_counts": sum_citation[idx]})
                unstable_count+=1
                break
    unstable_rate = 1 - unstable_count/num_examples     
    return unstable_rate, unstable_count, unstable_examples

def compute_citation(pipe):
    """
    计算单次运行(pipe)中的引用情况。
    
    Args:
        pipe (list): 一次运行产生的所有 example 的列表。

    Returns:
        tuple: (fully_cited_count, incomplete_cited_ids)
               - fully_cited_count (int): 完全引用了所有源文档的样本数量。
               - incomplete_cited_ids (list): 未完全引用源文档的样本ID列表。
    """
    fully_cited_count = 0
    incomplete_cite_list = []
    
    for example in pipe:
        citations = count_citations(example)
        num_cited = sum(citations)
        total_docs = len(example.get('document', []))
        
        # 如果总文档数为0，我们认为它是“完全引用”的（因为没有需要引用的）
        if total_docs == 0:
            fully_cited_count += 1
            continue
        
        # 比较 “实际引用的独立文档数” 与 “总的可用文档数”
        if num_cited < total_docs:
            incomplete_cite_list.append({"flags":citations, "example": example})
        else:
            fully_cited_count += 1
    
    return fully_cited_count, incomplete_cite_list


def main(args):
    original_args_dict = deepcopy(args.__dict__) 
    with open(args.config_file, 'r') as f1:
        full_configs= json.loads(f1.read())
    
    # make sure all configs for all subtasks are supplied
    if not any(elem['subtask']=="content_selection" for elem in full_configs):
        raise Exception("must provide content_selection configs")
    if not set([elem['subtask'] for elem in full_configs]) in [{"content_selection", "clustering", "iterative_sentence_generation"}, {"content_selection", "fusion_in_context"}]:
        raise Exception('configs must be of the following subtasks: (1) "content_selection", "clustering", "iterative_sentence_generation"; or (2) "content_selection", "fusion_in_context"')
    
    # make sure all config files share the same split and setting
    all_splits, all_settings = [], []
    for elem in full_configs:
        with open(elem['config_file'], 'r') as f1:
            curr_configs = json.loads(f1.read())
            all_splits.append(curr_configs['split'])
            all_settings.append(curr_configs['setting'])
    if len(set(all_splits))!=1 or len(set(all_settings))!=1:
        raise Exception("all subtasks must have the same split (test/dev) and the same setting (MDS/LFQA)")

    # define and create outdir
    pipeline_subdir = "full_CoT_pipeline" if "fusion_in_context" in set([elem['subtask'] for elem in full_configs]) else "full_pipeline"
    outdir = args.outdir if args.outdir else  f"results/{all_splits[0]}/{all_settings[0]}/{args.test_target}"
    logging.info(f"saving results to {outdir}")
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist

    intermediate_outdir = os.path.join(outdir, args.task) # subdir with results of intermediate subtasks
    path = Path(intermediate_outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist

    
    if args.test_target == "all_counts":
        choose_idx =  [i for i in range(200)]
        content_selection_outdir = os.path.join(intermediate_outdir, "content_selection")
        logging.info("running content seletion:")
        pipe1 = run_subtask(full_configs=full_configs, 
                    subtask_name="content_selection", 
                    curr_outdir=content_selection_outdir, 
                    original_args_dict=original_args_dict, 
                    indir_alignments=args.indir_alignments,
                    choose_idx=choose_idx)
        # clustering
        clustering_outdir = os.path.join(intermediate_outdir, "clustering")
        logging.info("running clustering:")
        pipe2 = run_subtask(full_configs=full_configs, 
                    subtask_name="clustering", 
                    curr_outdir=clustering_outdir, 
                    original_args_dict=original_args_dict,
                    indir_alignments=os.path.join(content_selection_outdir, "pipeline_format_results.json")) # the alignments are the outputs of the previous subtask (content_selection)

        fully_cited_count, incomplete_cite_list = compute_citation(pipe2)
        results_to_save = {
            "all_counts": len(pipe2),
            "fully_cited_count": fully_cited_count,
            "incomplete_cite_list": incomplete_cite_list  # incomplete_cite_list 本身可能也是一个字典或列表
        }
        filename = 'filtered_question.json'
        os.makedirs(intermediate_outdir, exist_ok=True)
        filepath = os.path.join(intermediate_outdir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=4, ensure_ascii=False)
            print(f"结果已成功保存到: {filepath}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
        print(f"Fully Cited Count: {fully_cited_count}")
        # logging.info(f"Total Citations (All): {cite_all}, Citation Indices: {cite_idx}")
    
def count_citations(example):
    """
    接收一个 example 字典，返回一个标记列表，指示每个源文档是否被引用。
    
    Args:
        example (dict): 单个样本数据。

    Returns:
        list: 一个由0和1组成的列表。列表的索引对应源文档的索引，
              值为1表示被引用，0表示未被引用。
              例如 [1, 0, 1] 表示共有3个源文档，第0和第2个被引用了。
    """
    # 获取总的源文档数量
    num_documents = len(example.get('document', []))
    if num_documents == 0:
        return []

    # 初始化一个全为0的标记列表
    citation_flags = [0] * num_documents
    
    # 获取被引用的文档索引
    highlights = example.get('set_of_highlights_in_context', [])
    cited_indices = {int(h.get('documentFile')) for h in highlights if h.get('documentFile') is not None}
    
    # 将被引用过的位置标记为1
    for idx in cited_indices:
        if 0 <= idx < num_documents:
            citation_flags[idx] = 1
            
    return citation_flags


def return_res(id, query, rewritten_docs, args=None):
    """在属性先行管线中跑一条样本，返回引用标记与回答文本。"""
    args = args or pre_init([])
    base_dir = Path(__file__).resolve().parent
    original_args_dict = deepcopy(args.__dict__)
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = (base_dir / config_path).resolve()
    with open(config_path, 'r') as f1:
        full_configs= json.loads(f1.read())
    
    # make sure all configs for all subtasks are supplied
    if not any(elem['subtask']=="content_selection" for elem in full_configs):
        raise Exception("must provide content_selection configs")
    if not set([elem['subtask'] for elem in full_configs]) in [{"content_selection", "clustering"}, {"content_selection", "fusion_in_context"}]:
        raise Exception('configs must be of the following subtasks: (1) "content_selection", "clustering" or (2) "content_selection", "fusion_in_context"')
    
    # make sure all config files share the same split and setting
    all_splits, all_settings = [], []
    for elem in full_configs:
        cfg_path = Path(elem['config_file'])
        if not cfg_path.is_absolute():
            cfg_path = (base_dir / cfg_path).resolve()
        with open(cfg_path, 'r') as f1:
            curr_configs = json.loads(f1.read())
            all_splits.append(curr_configs['split'])
            all_settings.append(curr_configs['setting'])
        elem['config_file'] = str(cfg_path)
    if len(set(all_splits))!=1 or len(set(all_settings))!=1:
        raise Exception("all subtasks must have the same split (test/dev) and the same setting (MDS/LFQA)")

    # define and create outdir
    pipeline_subdir = "full_CoT_pipeline" if "fusion_in_context" in set([elem['subtask'] for elem in full_configs]) else "full_pipeline"
    outdir = args.outdir if args.outdir else  f"results/{all_splits[0]}/{all_settings[0]}/{pipeline_subdir}"
    # 确保 outdir 是绝对路径，避免后续路径解析错误
    outdir = str(Path(outdir).resolve())
    logging.info(f"saving results to {outdir}")
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist

    intermediate_outdir = os.path.join(outdir, "itermediate_results") # subdir with results of intermediate subtasks
    path = Path(intermediate_outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist
    # content selection
    content_selection_outdir = os.path.join(intermediate_outdir, "content_selection")
    input_output_path = os.path.join(intermediate_outdir, "pipeline_format_results.json")
    data = {
        "id": id,
        "document": [{"raw_text": doc, "url": "", "cleaned_text": doc} for doc in rewritten_docs],
        "question": query,
        "set_of_highlights_in_context": [],
    }
    with open(input_output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    logging.info("\n=== 开始 Content Selection 阶段 ===")
    logging.info(f"输入查询: {query[:100]}...")
    logging.info(f"输入文档数量: {len(rewritten_docs)}")
    logging.info(f"输入文档详情: {[(i+1, doc[:50]+'...') for i, doc in enumerate(rewritten_docs)]}")
    logging.info(f"Content Selection 输出目录: {content_selection_outdir}")
    
    run_subtask(
        full_configs=full_configs,
        subtask_name="content_selection",
        curr_outdir=content_selection_outdir,
        original_args_dict=original_args_dict,
        indir_alignments=input_output_path,
        choose_idx=[0],
        data_num=1,
    )
    
    logging.info("✅ Content Selection 阶段完成")
    final_alignment_path = os.path.join(content_selection_outdir, "pipeline_format_results.json")
    subtasks = {elem['subtask'] for elem in full_configs}

    if "clustering" in subtasks:
        clustering_outdir = os.path.join(intermediate_outdir, "clustering")
        logging.info("\n=== 开始 Clustering 阶段 ===")
        logging.info(f"Clustering 输出目录: {clustering_outdir}")
        logging.info(f"使用 Content Selection 结果路径: {final_alignment_path}")
        
        run_subtask(
            full_configs=full_configs,
            subtask_name="clustering",
            curr_outdir=clustering_outdir,
            original_args_dict=original_args_dict,
            indir_alignments=final_alignment_path,
        )
        
        logging.info("✅ Clustering 阶段完成")
        final_alignment_path = os.path.join(clustering_outdir, "pipeline_format_results.json")

    if "iterative_sentence_generation" in subtasks:
        # logging.info("\n=== 开始 Iterative Sentence Generation 阶段 ===")
        # logging.info(f"使用 Clustering 结果路径: {final_alignment_path}")
        
        # run_subtask(
        #     full_configs=full_configs,
        #     subtask_name="iterative_sentence_generation",
        #     curr_outdir=outdir,
        #     original_args_dict=original_args_dict,
        #     indir_alignments=final_alignment_path,
        # )
        
        logging.info("✅ Iterative Sentence Generation 阶段不做")
        # final_alignment_path = os.path.join(outdir, "pipeline_format_results.json")
    elif "fusion_in_context" in subtasks:
        # logging.info("\n=== 开始 Fusion in Context 阶段 ===")
        # logging.info(f"使用 Clustering 结果路径: {final_alignment_path}")
        
        # run_subtask(
        #     full_configs=full_configs,
        #     subtask_name="fusion_in_context",
        #     curr_outdir=outdir,
        #     original_args_dict=original_args_dict,
        #     indir_alignments=final_alignment_path,
        # )
        
        logging.info("✅ Fusion in Context 阶段不做")
        # final_alignment_path = os.path.join(outdir, "pipeline_format_results.json")

    logging.info("\n=== 开始生成最终答案阶段 ===")
    logging.info(f"读取最终结果路径: {final_alignment_path}")
    
    parsed_results = []
    with open(final_alignment_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed_results.append(json.loads(line))

    if not parsed_results:
        raise ValueError("未能生成 pipeline_format_results.json")

    pipe2 = parsed_results[0]
    citations = count_citations(pipe2)
    answer_prompt = pipe2.get("response", "")
    highlight_set = pipe2.get("set_of_highlights_in_context", [])
    
    logging.info(f"生成的答案: {answer_prompt[:150]}...")
    logging.info(f"引用计数: {citations}")
    logging.info(f"高亮集合大小: {len(highlight_set)}")
    logging.info(f"高亮详情: {[(h.get('doc_idx', 0)+1, h.get('highlight_text', '')[:50]+'...') for h in highlight_set[:3]]}{'...' if len(highlight_set) > 3 else ''}")
    logging.info("✅ 生成最终答案阶段完成")

    res = {
        "citations": citations,
        "answer_prompt": answer_prompt,
        "highlight_set": highlight_set
        }
    logging.info("\n=== AttrEvaluator 完整流程结束 ===")
    return res

def pre_init(arg_list=None):
    argparser = argparse.ArgumentParser(description="")
    base_dir = Path(__file__).resolve().parent
    default_config = str(base_dir / "configs/test/LFQA/full_pipeline.json")
    argparser.add_argument('--config-file', type=str, default=default_config, help='path to json config file.')
    argparser.add_argument('--task', type=str, default="task_0", help='path to json config file.')
    argparser.add_argument('--original-pipe', type=str, default=str(base_dir / "configs/test/LFQA/full_pipeline.json"), help='path to json config file.')
    argparser.add_argument('--test-target', type=str, default="all_counts", help='the test target to use (default: all_counts, gpt_stability, generate_filtered_dataset, attack)')
    argparser.add_argument('-o', '--outdir', type=str, default=None, help='path to output csv.')
    argparser.add_argument('--indir-alignments', type=str, default=None, help='path to json file with alignments (if nothing is passed - goes to default under data/{setting}/{split}.json).')
    argparser.add_argument('--indir-prompt', type=str, default=None, help='path to json file with the prompt structure and ICL examples (if nothing is passed - goes to default under prompts/{setting}.json).')
    argparser.add_argument('--model-name', type=str, default="gpt-4.1-mini", help='model name')
    argparser.add_argument('--n-demos', type=int, default=2, help='number of ICL examples (default 2)')
    argparser.add_argument('--compete-times', type=int, default=3, help='number of gpt compete tim (default 4)')
    argparser.add_argument('--data-num', type=int, default=50, help='number of data samples to use (default 50)')
    argparser.add_argument('--num-retries', type=int, default=1, help='number of retries of running the model.')
    argparser.add_argument('--temperature', type=float, default=0.2, help='temperature of generation')
    argparser.add_argument('--debugging', action='store_true', default=False, help='if debugging mode.')
    argparser.add_argument('--merge-cross-sents-highlights', action='store_true', default=False, help='whether to merge consecutive highlights that span across several sentences.')    
    argparser.add_argument('--CoT', action='store_true', default=False, help='whether to use a CoT approach (relevant for FiC and clustering).')    
    argparser.add_argument('--cut-surplus', action='store_true', default=False, help='whether to cut surplus text from prompts (in subtask with given highlights - everything after last highlight, and in tasks without - last prct_surplus sentences).')
    argparser.add_argument('--prct-surplus', type=float, default=None, help='for subtasks without given highlights (e.g. content_selection, e2e_only_setting, or ALCE) - what percentage of top document sents to drop in cases when the prompts are too long.')
    argparser.add_argument('--always-with-question', action='store_true', default=False, help='relevant for LFQA - whether to always add the question (also to clustering and FiC)')
    argparser.add_argument('--num-demo-changes', type=int, default=4, help='number of changing demos when the currently-chosen set of demos returns an ERROR.')
    argparser.add_argument('--rerun', action='store_true', default=False, help='if need to rerun on instances that had errors')
    argparser.add_argument('--rerun-path', type=str, default=None, help='path to rerun on (where the results are)')
    argparser.add_argument('--rerun-n-demos', type=int, default=None, help='new n_demos for rerun in cases when the current n_demos doesnt work.')
    argparser.add_argument('--rerun-temperature', type=float, default=None, help='new temperature for rerun in cases when the current temperature doesnt work.')
    argparser.add_argument('--no-prefix', action='store_true', default=False, help='ablation study where the prefix is not add.')
    argparser.add_argument('--find-leak', action='store_true', default=False, help='if make leak detection.')
    argparser.add_argument('--choose-idx', type=str, default=None, help='comma-separated list of instance indices to run on (e.g. "0,2,5"). If set, only those indices will be kept from the alignments.')
    argparser.add_argument('--idx-to-attack', type=int, default=0, help='index of the instance to attack')
    args = argparser.parse_args(arg_list)
    return args 


if __name__ == "__main__":
    # 需要被加入args的："filtered_dataset_100.json"存档数据集读入路径，现在服用indir_alignments，运行模式， 攻击模式
    # 直接运行测试需要设置PYTHONPATH
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    from agent_geo.baselines.attr_evaluator.run_dataset import pre_init, return_res
    
    # 初始化参数
    config_file = Path(__file__).resolve().parent / "configs" / "test" / "LFQA" / "full_pipeline.json"
    args = pre_init(['--config-file', str(config_file)])
    
    # 测试数据
    doc = ["Document 1 text about AI.", "Document 2 text about machine learning.", "Document 3 text about deep learning."]
    query = "What is the main topic discussed in the documents?"
    
    try:
        print("=== 测试 run_dataset.py 修改 ===")
        res = return_res(id=0, query=query, rewritten_docs=doc, args=args)
        print(f"测试结果: {res}")
        print("=== 测试成功 ===")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
