"""
HTML Content Modification System - Complete Rewrite Version

Features:
1. Intelligently extract structured content from complex HTML
2. Use GPT-4o-mini to analyze the mapping between content and HTML
3. Support add, modify, and delete sentence operations
4. Return the modified complete HTML

Author: AI Assistant
Date: 2025-12-27
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
# HTML processing
from bs4 import BeautifulSoup, NavigableString

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Model Definitions
# ============================================================================

class Change(BaseModel):
    """Single change operation"""
    type: str = Field(..., description="Change type: 'add', 'modify', or 'delete'")
    anchor_sentence: str = Field(..., description="The sentence before the change (for locating in HTML)")
    new_sentence: Optional[str] = Field(None, description="New sentence for 'add' or 'modify' operations")
    deleted_sentence: Optional[str] = Field(None, description="Deleted sentence for 'delete' operations")
    confidence: float = Field(default=0.9, description="Confidence score of this change (0-1)")


class ChangesOutput(BaseModel):
    """List of changes returned by LLM"""
    changes: List[Change] = Field(..., description="List of identified changes")
    summary: Optional[str] = Field(None, description="Summary of all changes")



# ============================================================================
# HTML Modification Engine
# ============================================================================

class HTMLModificationEngine:
    """
    Use LLM to analyze changes and apply modifications in HTML.
    """

    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0):
        """Initialize LLM"""
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=ChangesOutput)
    
    def analyze_changes(self, original_text: str, new_content: str) -> List[Change]:
        """
        Use LLM to analyze differences between original text and new content.

        Args:
            original_text: Original text extracted from HTML
            new_content: New content text

        Returns:
            List of changes
        """
        logger.info("Starting LLM content difference analysis...")

        # Truncate text to avoid token limit
        max_length = 8000
        original_snippet = original_text[:max_length]
        new_snippet = new_content[:max_length]

        prompt = f"""You are a content comparison expert. I have text extracted from the original HTML page and new content text, and I need you to identify changes.

Tasks:
1. Identify new sentences that are in "new content" but not in "original text" (type: "add").
2. Identify sentences in "new content" that have been significantly modified compared to "original text" (type: "modify").
3. Identify sentences that are in "original text" but not in "new content" (type: "delete").
4. For each change, find the sentence immediately preceding the change in "original text" as the "anchor_sentence".
5. anchor_sentence must exist completely in the original text.

Important rules:
- For "add" and "modify", provide "new_sentence".
- For "delete", provide "deleted_sentence".
- anchor_sentence should be the previous complete sentence, used for locating in HTML.
- If no suitable anchor_sentence can be found, use the closest preceding text.

Original text:
{original_snippet}

New content:
{new_snippet}

Please output in JSON format, following this structure:
{self.parser.get_format_instructions()}
"""

        try:
            response = self.llm.invoke(prompt)
            parsed = self.parser.parse(response.content)
            changes = parsed.changes
            logger.info(f"Identified {len(changes)} changes")
            return changes
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            logger.error(f"LLM response: {response.content if 'response' in locals() else 'N/A'}")
            return []
    
    def apply_changes_to_html(self, html: str, changes: List[Change]) -> str:
        """
        Apply changes to HTML.

        Strategy:
        1. For "add": Insert new sentence after anchor_sentence (wrapped in special span)
        2. For "modify": Find and replace the corresponding sentence (wrapped in special span)
        3. For "delete": Find and mark the corresponding sentence as deleted (wrapped in strikethrough span)

        Args:
            html: Original HTML
            changes: List of changes

        Returns:
            Modified HTML
        """
        logger.info(f"Starting to apply {len(changes)} changes to HTML...")

        soup = BeautifulSoup(html, 'html.parser')

        # Process in reverse order by position to avoid position offset
        for change in sorted(changes, key=lambda c: c.position if hasattr(c, 'position') else 0, reverse=True):
            try:
                if change.type == "add":
                    self._apply_add(soup, change)
                elif change.type == "modify":
                    self._apply_modify(soup, change)
                elif change.type == "delete":
                    self._apply_delete(soup, change)
            except Exception as e:
                logger.warning(f"Failed to apply change ({change.type}): {e}")

        return str(soup.prettify())
    
    def _apply_add(self, soup: BeautifulSoup, change: Change) -> None:
        """Apply "add" change: Insert new sentence after anchor"""
        anchor = change.anchor_sentence.strip()
        new_text = change.new_sentence.strip()

        if not anchor or not new_text:
            return

        # Find anchor sentence
        target_node = self._find_text_node(soup, anchor)

        if target_node:
            # Create new span element (yellow background, indicating addition)
            new_span = soup.new_tag("span")
            new_span['class'] = "html-modifier-added"
            new_span['style'] = "background-color: #FFFF00; color: #000000; font-weight: bold; border: 2px solid #FFA500; padding: 3px; margin: 2px; display: inline-block;"
            new_span.string = f"[Added] {new_text}"

            # Insert after anchor node
            if target_node.parent:
                target_node.insert_after(new_span)
                logger.info(f"Added: {new_text[:50]}...")
            else:
                logger.warning(f"Cannot locate anchor's parent node")
        else:
            logger.warning(f"Anchor sentence not found: {anchor[:50]}...")
    
    def _apply_modify(self, soup: BeautifulSoup, change: Change) -> None:
        """Apply "modify" change: Replace the corresponding sentence"""
        anchor = change.anchor_sentence.strip()
        new_text = change.new_sentence.strip()

        if not anchor or not new_text:
            return

        # Find the sentence to modify (usually the sentence after anchor)
        # Here we use a smarter approach: find the element containing anchor, then modify content after it
        target_node = self._find_text_node(soup, anchor)

        if target_node:
            # Find next text node or element
            next_node = target_node.find_next(string=True)

            if next_node and next_node.parent:
                # Create modified span (blue background, indicating modification)
                modified_span = soup.new_tag("span")
                modified_span['class'] = "html-modifier-modified"
                modified_span['style'] = "background-color: #87CEEB; color: #000000; font-weight: bold; border: 2px solid #0000FF; padding: 3px; margin: 2px; display: inline-block;"
                modified_span.string = f"[Modified] {new_text}"

                next_node.replace_with(modified_span)
                logger.info(f"Modified: {new_text[:50]}...")
            else:
                # If next node not found, insert directly after anchor
                new_span = soup.new_tag("span")
                new_span['class'] = "html-modifier-modified"
                new_span['style'] = "background-color: #87CEEB; color: #000000; font-weight: bold; border: 2px solid #0000FF; padding: 3px; margin: 2px; display: inline-block;"
                new_span.string = f"[Modified] {new_text}"

                if target_node.parent:
                    target_node.replace_with(new_span)
                    logger.info(f"Modified (inserted): {new_text[:50]}...")
        else:
            logger.warning(f"Anchor sentence not found for modification: {anchor[:50]}...")
    
    def _apply_delete(self, soup: BeautifulSoup, change: Change) -> None:
        """Apply "delete" change: Mark the deleted sentence"""
        deleted_text = change.deleted_sentence.strip()

        if not deleted_text:
            return

        # Find the deleted sentence
        target_node = self._find_text_node(soup, deleted_text)

        if target_node:
            # Create deletion marker span (red background + strikethrough)
            del_span = soup.new_tag("span")
            del_span['class'] = "html-modifier-deleted"
            del_span['style'] = "background-color: #FF6B6B; color: #FFFFFF; font-weight: bold; text-decoration: line-through; border: 2px solid #FF0000; padding: 3px; margin: 2px; display: inline-block;"
            del_span.string = f"[Deleted] {deleted_text}"

            target_node.replace_with(del_span)
            logger.info(f"Deleted: {deleted_text[:50]}...")
        else:
            logger.warning(f"Deleted sentence not found: {deleted_text[:50]}...")
    
    def _find_text_node(self, soup: BeautifulSoup, text: str) -> Optional[Any]:
        """
        Find a node containing the specified text in HTML.
        Use fuzzy matching to handle whitespace and formatting differences.
        """
        # Normalize search text
        normalized_search = text.strip().lower()

        # First try exact match
        for node in soup.find_all(string=True):
            if isinstance(node, NavigableString) and not isinstance(node, str):
                continue

            node_text = str(node).strip().lower()

            # Exact match
            if node_text == normalized_search:
                return node

            # Contains match (for partial sentences)
            if len(normalized_search) > 20 and normalized_search in node_text:
                return node

        # If not found, try fuzzy matching (find nodes containing keywords)
        keywords = normalized_search.split()[:3]  # Take first 3 keywords

        for node in soup.find_all(string=True):
            if isinstance(node, NavigableString) and not isinstance(node, str):
                continue

            node_text = str(node).strip().lower()

            if all(keyword in node_text for keyword in keywords):
                return node

        return None


# ============================================================================
# Main Controller
# ============================================================================

class HTMLContentModifier:
    """
    Main controller: Coordinates the entire HTML modification workflow.
    """

    OUTPUT_DIR = "refined_pages"

    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0):
        """Initialize modifier"""
        self.engine = HTMLModificationEngine(model=model, temperature=temperature)
        self.parser = HtmlParser()

        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
            logger.info(f"Created output directory: {self.OUTPUT_DIR}")
    
    def _get_output_path(self, hint: Optional[str] = None) -> str:
        """Generate output file path"""
        if not hint:
            filename = f"modified_{uuid.uuid4().hex[:8]}.html"
        else:
            # Use MD5 hash of hint as filename
            filename = f"modified_{hashlib.md5(hint.encode()).hexdigest()}.html"

        return os.path.join(self.OUTPUT_DIR, filename)
    
    def modify(self, html: str, content: str, filename_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete HTML modification workflow.

        Args:
            html: Original HTML content
            content: New content text
            filename_hint: Output filename hint

        Returns:
            Dictionary containing modification results
        """
        logger.info("=" * 60)
        logger.info("Starting HTML content modification workflow")
        logger.info("=" * 60)

        if not html or not html.strip():
            logger.error("HTML content is empty")
            return {
                "status": "error",
                "message": "HTML content is empty",
                "html": html
            }

        if not content or not content.strip():
            logger.error("New content is empty")
            return {
                "status": "error",
                "message": "New content is empty",
                "html": html
            }

        try:
            # 1. Extract original text from HTML
            logger.info("Step 1: Extracting original text...")
            original_text = self.parser.to_clean_text_bs4(html)
            logger.info(f"Extracted {len(original_text)} characters of text")

            # 2. Use LLM to analyze changes
            logger.info("Step 2: Using LLM to analyze changes...")
            changes = self.engine.analyze_changes(original_text, content)

            if not changes:
                logger.warning("No changes identified")
                return {
                    "status": "no_changes",
                    "message": "No changes identified",
                    "changes_count": 0,
                    "html": html
                }

            logger.info(f"Identified {len(changes)} changes")

            # 3. Apply changes to HTML
            logger.info("Step 3: Applying changes to HTML...")
            modified_html = self.engine.apply_changes_to_html(html, changes)

            # 4. Save modified HTML
            logger.info("Step 4: Saving modified HTML...")
            output_path = self._get_output_path(filename_hint)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(modified_html)

            abs_path = str(Path(output_path).resolve())
            logger.info(f"File saved: {abs_path}")

            # 5. Return results
            logger.info("=" * 60)
            logger.info("HTML modification completed")
            logger.info("=" * 60)

            return {
                "status": "success",
                "message": "HTML modification successful",
                "changes_count": len(changes),
                "changes": [c.dict() for c in changes],
                "local_file_path": abs_path,
                "html": modified_html
            }

        except Exception as e:
            logger.error(f"Modification process error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Modification process error: {str(e)}",
                "html": html
            }


# ============================================================================
# Export Functions
# ============================================================================

def modify_html(html: str, content: str, filename_hint: str = "doc") -> str:
    """
    Main function to modify HTML content.

    Args:
        html: Original HTML content
        content: New content text
        filename_hint: Output filename hint

    Returns:
        JSON formatted result string
    """
    modifier = HTMLContentModifier()
    result = modifier.modify(html, content, filename_hint)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    # Check API Key
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("Please set environment variable OPENAI_API_KEY")
        exit(1)

    print("\n" + "=" * 70)
    print("HTML Content Modification System - Complex Test Cases")
    print("=" * 70 + "\n")

    # ========================================================================
    # Test Case 1: Basic Test (Add, Modify, Delete)
    # ========================================================================
    print("\n[Test Case 1] Basic Functionality Test (Add, Modify, Delete)")
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

    print(f"Status: {result1['status']}")
    print(f"Changes count: {result1.get('changes_count', 0)}")
    if result1.get('changes'):
        print(f"Change details:")
        for i, change in enumerate(result1['changes'], 1):
            print(f"  {i}. Type: {change['type']}")
            print(f"     Anchor: {change['anchor_sentence'][:50]}...")
            if change.get('new_sentence'):
                print(f"     New sentence: {change['new_sentence'][:50]}...")
            if change.get('deleted_sentence'):
                print(f"     Deleted sentence: {change['deleted_sentence'][:50]}...")

    if result1['status'] == 'success':
        print(f"✓ File saved: {result1['local_file_path']}")

    # ========================================================================
    # Test Case 2: Complex Web Page Structure
    # ========================================================================
    print("\n\n[Test Case 2] Complex Web Page Structure Test")
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

    print(f"Status: {result2['status']}")
    print(f"Changes count: {result2.get('changes_count', 0)}")
    if result2.get('changes'):
        print(f"Change details:")
        for i, change in enumerate(result2['changes'], 1):
            print(f"  {i}. Type: {change['type']}")
            print(f"     Anchor: {change['anchor_sentence'][:50]}...")
            if change.get('new_sentence'):
                print(f"     New sentence: {change['new_sentence'][:50]}...")
            if change.get('deleted_sentence'):
                print(f"     Deleted sentence: {change['deleted_sentence'][:50]}...")

    if result2['status'] == 'success':
        print(f"✓ File saved: {result2['local_file_path']}")

    # ========================================================================
    # Test Case 3: Large Text Modification
    # ========================================================================
    print("\n\n[Test Case 3] Large Text Modification Test")
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

    print(f"Status: {result3['status']}")
    print(f"Changes count: {result3.get('changes_count', 0)}")
    if result3.get('changes'):
        print(f"First 5 changes:")
        for i, change in enumerate(result3['changes'][:5], 1):
            print(f"  {i}. Type: {change['type']}")
            print(f"     Anchor: {change['anchor_sentence'][:50]}...")
            if change.get('new_sentence'):
                print(f"     New sentence: {change['new_sentence'][:50]}...")

    if result3['status'] == 'success':
        print(f"✓ File saved: {result3['local_file_path']}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("All tests completed")
    print("=" * 70)
    print(f"\nOutput directory: {os.path.abspath(HTMLContentModifier.OUTPUT_DIR)}")
    print("\nYou can open the generated HTML files in a browser to view the modification results:")
    print("- Yellow background = Added content")
    print("- Blue background = Modified content")
    print("- Red background + strikethrough = Deleted content")
