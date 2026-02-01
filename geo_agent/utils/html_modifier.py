"""
HTML 内容修改系统 - 完整重写版本

功能：
1. 从复杂 HTML 中智能提取结构化内容
2. 使用 GPT-4o-mini 分析 content 与 HTML 的对应关系
3. 支持添加、修改、删除句子操作
4. 返回修改后的完整 HTML

作者: AI Assistant
日期: 2025-12-27
"""

import json
import logging
import os
import hashlib
import uuid
from typing import Dict, Optional, List, Any
from pathlib import Path

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from geo_agent.utils.html_parser import HtmlParser
# HTML 处理
from bs4 import BeautifulSoup, NavigableString

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型定义
# ============================================================================

class Change(BaseModel):
    """单个变更操作"""
    type: str = Field(..., description="Change type: 'add', 'modify', or 'delete'")
    anchor_sentence: str = Field(..., description="The sentence before the change (for locating in HTML)")
    new_sentence: Optional[str] = Field(None, description="New sentence for 'add' or 'modify' operations")
    deleted_sentence: Optional[str] = Field(None, description="Deleted sentence for 'delete' operations")
    confidence: float = Field(default=0.9, description="Confidence score of this change (0-1)")


class ChangesOutput(BaseModel):
    """LLM 返回的变更列表"""
    changes: List[Change] = Field(..., description="List of identified changes")
    summary: Optional[str] = Field(None, description="Summary of all changes")



# ============================================================================
# HTML 修改引擎
# ============================================================================

class HTMLModificationEngine:
    """
    使用 LLM 分析变更，并在 HTML 中应用修改。
    """
    
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0):
        """初始化 LLM"""
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=ChangesOutput)
    
    def analyze_changes(self, original_text: str, new_content: str) -> List[Change]:
        """
        使用 LLM 分析原始文本和新内容的差异。
        
        Args:
            original_text: 从 HTML 提取的原始文本
            new_content: 新的内容文本
        
        Returns:
            变更列表
        """
        logger.info("开始使用 LLM 分析内容差异...")
        
        # 截断过长的文本以避免 token 超限
        max_length = 8000
        original_snippet = original_text[:max_length]
        new_snippet = new_content[:max_length]
        
        prompt = f"""你是一个内容对比专家。我有原始 HTML 页面提取的文本和新的内容文本，需要你识别变更。

任务：
1. 识别在"新内容"中但不在"原始文本"中的新增句子（type: "add"）。
2. 识别"新内容"中相比"原始文本"有明显修改的句子（type: "modify"）。
3. 识别在"原始文本"中但不在"新内容"中的删除句子（type: "delete"）。
4. 对于每个变更，找到"原始文本"中紧邻该变更前面的句子作为"anchor_sentence"。
5. anchor_sentence 必须完全存在于原始文本中。

重要规则：
- 对于 "add" 和 "modify"，提供 "new_sentence"。
- 对于 "delete"，提供 "deleted_sentence"。
- anchor_sentence 应该是前一个完整的句子，用于在 HTML 中定位。
- 如果找不到合适的 anchor_sentence，使用最接近的前文。

原始文本：
{original_snippet}

新内容：
{new_snippet}

请以 JSON 格式输出，遵循以下结构：
{self.parser.get_format_instructions()}
"""
        
        try:
            response = self.llm.invoke(prompt)
            parsed = self.parser.parse(response.content)
            changes = parsed.changes
            logger.info(f"识别到 {len(changes)} 个变更")
            return changes
        except Exception as e:
            logger.error(f"LLM 分析失败: {e}")
            logger.error(f"LLM 响应: {response.content if 'response' in locals() else 'N/A'}")
            return []
    
    def apply_changes_to_html(self, html: str, changes: List[Change]) -> str:
        """
        将变更应用到 HTML 中。
        
        策略：
        1. 对于 "add"：在 anchor_sentence 后插入新句子（用特殊 span 包装）
        2. 对于 "modify"：找到对应句子并替换（用特殊 span 包装）
        3. 对于 "delete"：找到对应句子并标记为删除（用删除线 span 包装）
        
        Args:
            html: 原始 HTML
            changes: 变更列表
        
        Returns:
            修改后的 HTML
        """
        logger.info(f"开始应用 {len(changes)} 个变更到 HTML...")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 按位置倒序处理，避免位置偏移
        for change in sorted(changes, key=lambda c: c.position if hasattr(c, 'position') else 0, reverse=True):
            try:
                if change.type == "add":
                    self._apply_add(soup, change)
                elif change.type == "modify":
                    self._apply_modify(soup, change)
                elif change.type == "delete":
                    self._apply_delete(soup, change)
            except Exception as e:
                logger.warning(f"应用变更失败 ({change.type}): {e}")
        
        return str(soup.prettify())
    
    def _apply_add(self, soup: BeautifulSoup, change: Change) -> None:
        """应用 "add" 变更：在 anchor 后插入新句子"""
        anchor = change.anchor_sentence.strip()
        new_text = change.new_sentence.strip()
        
        if not anchor or not new_text:
            return
        
        # 查找 anchor 句子
        target_node = self._find_text_node(soup, anchor)
        
        if target_node:
            # 创建新的 span 元素（黄色背景，表示新增）
            new_span = soup.new_tag("span")
            new_span['class'] = "html-modifier-added"
            new_span['style'] = "background-color: #FFFF00; color: #000000; font-weight: bold; border: 2px solid #FFA500; padding: 3px; margin: 2px; display: inline-block;"
            new_span.string = f"[新增] {new_text}"
            
            # 在 anchor 节点后插入
            if target_node.parent:
                target_node.insert_after(new_span)
                logger.info(f"已添加: {new_text[:50]}...")
            else:
                logger.warning(f"无法定位 anchor 的父节点")
        else:
            logger.warning(f"未找到 anchor 句子: {anchor[:50]}...")
    
    def _apply_modify(self, soup: BeautifulSoup, change: Change) -> None:
        """应用 "modify" 变更：替换对应的句子"""
        anchor = change.anchor_sentence.strip()
        new_text = change.new_sentence.strip()
        
        if not anchor or not new_text:
            return
        
        # 查找需要修改的句子（通常是 anchor 后的下一个句子）
        # 这里我们采用更智能的方法：查找包含 anchor 的元素，然后修改其后的内容
        target_node = self._find_text_node(soup, anchor)
        
        if target_node:
            # 查找下一个文本节点或元素
            next_node = target_node.find_next(string=True)
            
            if next_node and next_node.parent:
                # 创建修改后的 span（蓝色背景，表示修改）
                modified_span = soup.new_tag("span")
                modified_span['class'] = "html-modifier-modified"
                modified_span['style'] = "background-color: #87CEEB; color: #000000; font-weight: bold; border: 2px solid #0000FF; padding: 3px; margin: 2px; display: inline-block;"
                modified_span.string = f"[修改] {new_text}"
                
                next_node.replace_with(modified_span)
                logger.info(f"已修改: {new_text[:50]}...")
            else:
                # 如果找不到下一个节点，直接在 anchor 后插入
                new_span = soup.new_tag("span")
                new_span['class'] = "html-modifier-modified"
                new_span['style'] = "background-color: #87CEEB; color: #000000; font-weight: bold; border: 2px solid #0000FF; padding: 3px; margin: 2px; display: inline-block;"
                new_span.string = f"[修改] {new_text}"
                
                if target_node.parent:
                    target_node.replace_with(new_span)
                    logger.info(f"已修改（插入）: {new_text[:50]}...")
        else:
            logger.warning(f"未找到 anchor 句子用于修改: {anchor[:50]}...")
    
    def _apply_delete(self, soup: BeautifulSoup, change: Change) -> None:
        """应用 "delete" 变更：标记删除的句子"""
        deleted_text = change.deleted_sentence.strip()
        
        if not deleted_text:
            return
        
        # 查找被删除的句子
        target_node = self._find_text_node(soup, deleted_text)
        
        if target_node:
            # 创建删除标记 span（红色背景 + 删除线）
            del_span = soup.new_tag("span")
            del_span['class'] = "html-modifier-deleted"
            del_span['style'] = "background-color: #FF6B6B; color: #FFFFFF; font-weight: bold; text-decoration: line-through; border: 2px solid #FF0000; padding: 3px; margin: 2px; display: inline-block;"
            del_span.string = f"[删除] {deleted_text}"
            
            target_node.replace_with(del_span)
            logger.info(f"已删除: {deleted_text[:50]}...")
        else:
            logger.warning(f"未找到删除句子: {deleted_text[:50]}...")
    
    def _find_text_node(self, soup: BeautifulSoup, text: str) -> Optional[Any]:
        """
        在 HTML 中查找包含指定文本的节点。
        使用模糊匹配以处理空格和格式差异。
        """
        # 规范化搜索文本
        normalized_search = text.strip().lower()
        
        # 首先尝试精确匹配
        for node in soup.find_all(string=True):
            if isinstance(node, NavigableString) and not isinstance(node, str):
                continue
            
            node_text = str(node).strip().lower()
            
            # 精确匹配
            if node_text == normalized_search:
                return node
            
            # 包含匹配（用于部分句子）
            if len(normalized_search) > 20 and normalized_search in node_text:
                return node
        
        # 如果没有找到，尝试模糊匹配（查找包含关键词的节点）
        keywords = normalized_search.split()[:3]  # 取前3个关键词
        
        for node in soup.find_all(string=True):
            if isinstance(node, NavigableString) and not isinstance(node, str):
                continue
            
            node_text = str(node).strip().lower()
            
            if all(keyword in node_text for keyword in keywords):
                return node
        
        return None


# ============================================================================
# 主控制器
# ============================================================================

class HTMLContentModifier:
    """
    主控制器：协调整个 HTML 修改流程。
    """
    
    OUTPUT_DIR = "refined_pages"
    
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0):
        """初始化修改器"""
        self.engine = HTMLModificationEngine(model=model, temperature=temperature)
        self.parser = HtmlParser()
        
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
            logger.info(f"创建输出目录: {self.OUTPUT_DIR}")
    
    def _get_output_path(self, hint: Optional[str] = None) -> str:
        """生成输出文件路径"""
        if not hint:
            filename = f"modified_{uuid.uuid4().hex[:8]}.html"
        else:
            # 使用 hint 的 MD5 哈希作为文件名
            filename = f"modified_{hashlib.md5(hint.encode()).hexdigest()}.html"
        
        return os.path.join(self.OUTPUT_DIR, filename)
    
    def modify(self, html: str, content: str, filename_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整的 HTML 修改流程。
        
        Args:
            html: 原始 HTML 内容
            content: 新的内容文本
            filename_hint: 输出文件名提示
        
        Returns:
            包含修改结果的字典
        """
        logger.info("=" * 60)
        logger.info("开始 HTML 内容修改流程")
        logger.info("=" * 60)
        
        if not html or not html.strip():
            logger.error("HTML 内容为空")
            return {
                "status": "error",
                "message": "HTML 内容为空",
                "html": html
            }
        
        if not content or not content.strip():
            logger.error("新内容为空")
            return {
                "status": "error",
                "message": "新内容为空",
                "html": html
            }
        
        try:
            # 1. 从 HTML 提取原始文本
            logger.info("步骤 1: 提取原始文本...")
            original_text = self.parser.to_clean_text_bs4(html)
            logger.info(f"提取到 {len(original_text)} 个字符的文本")
            
            # 2. 使用 LLM 分析变更
            logger.info("步骤 2: 使用 LLM 分析变更...")
            changes = self.engine.analyze_changes(original_text, content)
            
            if not changes:
                logger.warning("未识别到任何变更")
                return {
                    "status": "no_changes",
                    "message": "未识别到任何变更",
                    "changes_count": 0,
                    "html": html
                }
            
            logger.info(f"识别到 {len(changes)} 个变更")
            
            # 3. 应用变更到 HTML
            logger.info("步骤 3: 应用变更到 HTML...")
            modified_html = self.engine.apply_changes_to_html(html, changes)
            
            # 4. 保存修改后的 HTML
            logger.info("步骤 4: 保存修改后的 HTML...")
            output_path = self._get_output_path(filename_hint)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(modified_html)
            
            abs_path = str(Path(output_path).resolve())
            logger.info(f"文件已保存: {abs_path}")
            
            # 5. 返回结果
            logger.info("=" * 60)
            logger.info("HTML 修改完成")
            logger.info("=" * 60)
            
            return {
                "status": "success",
                "message": "HTML 修改成功",
                "changes_count": len(changes),
                "changes": [c.dict() for c in changes],
                "local_file_path": abs_path,
                "html": modified_html
            }
        
        except Exception as e:
            logger.error(f"修改过程出错: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"修改过程出错: {str(e)}",
                "html": html
            }


# ============================================================================
# 导出函数
# ============================================================================

def modify_html(html: str, content: str, filename_hint: str = "doc") -> str:
    """
    修改 HTML 内容的主函数。
    
    Args:
        html: 原始 HTML 内容
        content: 新的内容文本
        filename_hint: 输出文件名提示
    
    Returns:
        JSON 格式的结果字符串
    """
    modifier = HTMLContentModifier()
    result = modifier.modify(html, content, filename_hint)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 检查 API Key
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("请设置环境变量 OPENAI_API_KEY")
        exit(1)
    
    print("\n" + "=" * 70)
    print("HTML 内容修改系统 - 复杂测试用例")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # 测试用例 1: 基础测试（添加、修改、删除）
    # ========================================================================
    print("\n【测试用例 1】基础功能测试（添加、修改、删除）")
    print("-" * 70)
    
    html_test1 = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Python 编程指南</title>
    </head>
    <body>
        <div class="container">
            <h1>Python 编程指南</h1>
            <p>Python 是一种高级编程语言，以其简洁易读的语法而闻名。</p>
            <p>它支持多种编程范式，包括面向对象编程和函数式编程。</p>
            <p>Python 在数据科学领域应用广泛。</p>
            <p>许多企业都在使用 Python 进行后端开发。</p>
            <h2>Python 的优势</h2>
            <p>Python 拥有庞大的标准库和第三方库生态。</p>
            <p>它的学习曲线相对平缓，适合初学者。</p>
        </div>
    </body>
    </html>
    """
    
    content_test1 = """
    Python 编程指南
    Python 是一种高级编程语言，以其简洁易读的语法而闻名。
    它支持多种编程范式，包括面向对象编程、函数式编程和响应式编程。
    Python 在人工智能和机器学习领域应用广泛。
    许多企业都在使用 Python 进行后端开发。
    Python 拥有强大的异步编程支持。
    Python 的优势
    Python 拥有庞大的标准库和第三方库生态。
    它的学习曲线相对平缓，适合初学者。
    """
    
    modifier = HTMLContentModifier()
    result1 = modifier.modify(html_test1, content_test1, "test_case_1")
    
    print(f"状态: {result1['status']}")
    print(f"变更数: {result1.get('changes_count', 0)}")
    if result1.get('changes'):
        print(f"变更详情:")
        for i, change in enumerate(result1['changes'], 1):
            print(f"  {i}. 类型: {change['type']}")
            print(f"     Anchor: {change['anchor_sentence'][:50]}...")
            if change.get('new_sentence'):
                print(f"     新句子: {change['new_sentence'][:50]}...")
            if change.get('deleted_sentence'):
                print(f"     删除句子: {change['deleted_sentence'][:50]}...")
    
    if result1['status'] == 'success':
        print(f"✓ 文件已保存: {result1['local_file_path']}")
    
    # ========================================================================
    # 测试用例 2: 复杂网页结构
    # ========================================================================
    print("\n\n【测试用例 2】复杂网页结构测试")
    print("-" * 70)
    
    html_test2 = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Web 开发技术栈</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .section { margin: 20px 0; }
        </style>
    </head>
    <body>
        <header>
            <h1>现代 Web 开发技术栈</h1>
            <p>Web 开发涉及多个层面的技术。</p>
        </header>
        <main>
            <section class="section">
                <h2>前端技术</h2>
                <p>HTML 是网页的基础结构。</p>
                <p>CSS 用于页面样式设计。</p>
                <p>JavaScript 提供交互功能。</p>
                <ul>
                    <li>React 是流行的前端框架。</li>
                    <li>Vue 提供渐进式的开发体验。</li>
                    <li>Angular 是企业级框架。</li>
                </ul>
            </section>
            <section class="section">
                <h2>后端技术</h2>
                <p>Node.js 使用 JavaScript 进行后端开发。</p>
                <p>Python 的 Django 框架功能完整。</p>
                <p>Java 的 Spring Boot 应用广泛。</p>
            </section>
            <section class="section">
                <h2>数据库</h2>
                <p>关系型数据库如 MySQL 和 PostgreSQL。</p>
                <p>NoSQL 数据库如 MongoDB 和 Redis。</p>
            </section>
        </main>
        <footer>
            <p>Web 开发是一个不断进化的领域。</p>
        </footer>
    </body>
    </html>
    """
    
    content_test2 = """
    现代 Web 开发技术栈
    Web 开发涉及多个层面的技术，从前端到后端再到数据库。
    前端技术
    HTML 是网页的基础结构，定义了内容和语义。
    CSS 用于页面样式设计和响应式布局。
    JavaScript 提供交互功能和动态行为。
    TypeScript 为 JavaScript 增加了类型安全。
    React 是流行的前端框架，由 Facebook 维护。
    Vue 提供渐进式的开发体验，学习曲线温和。
    Angular 是企业级框架，功能完整。
    后端技术
    Node.js 使用 JavaScript 进行后端开发，具有高性能。
    Python 的 Django 框架功能完整，内置 ORM。
    Java 的 Spring Boot 应用广泛，企业级支持强。
    Go 语言因其并发性能而受欢迎。
    数据库
    关系型数据库如 MySQL 和 PostgreSQL 提供 ACID 保证。
    NoSQL 数据库如 MongoDB 和 Redis 提供灵活的数据模型。
    Web 开发是一个不断进化的领域，需要持续学习。
    """
    
    result2 = modifier.modify(html_test2, content_test2, "test_case_2")
    
    print(f"状态: {result2['status']}")
    print(f"变更数: {result2.get('changes_count', 0)}")
    if result2.get('changes'):
        print(f"变更详情:")
        for i, change in enumerate(result2['changes'], 1):
            print(f"  {i}. 类型: {change['type']}")
            print(f"     Anchor: {change['anchor_sentence'][:50]}...")
            if change.get('new_sentence'):
                print(f"     新句子: {change['new_sentence'][:50]}...")
            if change.get('deleted_sentence'):
                print(f"     删除句子: {change['deleted_sentence'][:50]}...")
    
    if result2['status'] == 'success':
        print(f"✓ 文件已保存: {result2['local_file_path']}")
    
    # ========================================================================
    # 测试用例 3: 大量文本修改
    # ========================================================================
    print("\n\n【测试用例 3】大量文本修改测试")
    print("-" * 70)
    
    html_test3 = """
    <!DOCTYPE html>
    <html>
    <head><title>AI 技术发展</title></head>
    <body>
        <h1>人工智能技术发展历程</h1>
        <p>人工智能是计算机科学的一个重要分支。</p>
        <p>早期的 AI 主要基于符号推理。</p>
        <p>机器学习改变了 AI 的发展方向。</p>
        <p>深度学习推动了 AI 的快速发展。</p>
        <p>神经网络是深度学习的基础。</p>
        <p>Transformer 架构在自然语言处理中表现出色。</p>
        <p>大语言模型展示了 AI 的强大能力。</p>
    </body>
    </html>
    """
    
    content_test3 = """
    人工智能技术发展历程
    人工智能是计算机科学的一个重要分支，涉及多个学科领域。
    早期的 AI 主要基于符号推理和专家系统。
    机器学习改变了 AI 的发展方向，使系统能够从数据中学习。
    深度学习推动了 AI 的快速发展，特别是在计算机视觉领域。
    卷积神经网络在图像识别中取得突破性进展。
    神经网络是深度学习的基础，受到生物神经系统的启发。
    循环神经网络用于处理序列数据。
    Transformer 架构在自然语言处理中表现出色，成为现代 NLP 的标准。
    注意力机制是 Transformer 的核心创新。
    大语言模型展示了 AI 的强大能力，能够进行复杂的推理和生成。
    多模态模型融合了文本、图像和其他模态的信息。
    """
    
    result3 = modifier.modify(html_test3, content_test3, "test_case_3")
    
    print(f"状态: {result3['status']}")
    print(f"变更数: {result3.get('changes_count', 0)}")
    if result3.get('changes'):
        print(f"前 5 个变更:")
        for i, change in enumerate(result3['changes'][:5], 1):
            print(f"  {i}. 类型: {change['type']}")
            print(f"     Anchor: {change['anchor_sentence'][:50]}...")
            if change.get('new_sentence'):
                print(f"     新句子: {change['new_sentence'][:50]}...")
    
    if result3['status'] == 'success':
        print(f"✓ 文件已保存: {result3['local_file_path']}")
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("所有测试完成")
    print("=" * 70)
    print(f"\n输出目录: {os.path.abspath(HTMLContentModifier.OUTPUT_DIR)}")
    print("\n你可以在浏览器中打开生成的 HTML 文件查看修改效果：")
    print("- 黄色背景 = 新增内容")
    print("- 蓝色背景 = 修改内容")
    print("- 红色背景 + 删除线 = 删除内容")
