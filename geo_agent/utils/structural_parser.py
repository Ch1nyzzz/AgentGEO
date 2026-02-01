import uuid
import logging
import re
from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ContentChunk:
    """
    Deprecated: Retained for compatibility during refactoring.
    Can be removed once all references in optimizer.py are updated.
    """
    def __init__(self, index: int, elements: List[Dict]):
        self.index = index
        self._elements_data = elements
    
    @property
    def text(self) -> str:
        return "\n\n".join([e.get('text_content', '') for e in self._elements_data])
    
    @property
    def html(self) -> str:
        return "\n".join([e.get('original_html', '') for e in self._elements_data])

    @property
    def start_element_index(self) -> int:
         return 0 # Mock

class StructuralHtmlParser:
    def __init__(self, min_length: int = 50):
        """
        Initialize the parser.
        :param min_length: Minimum length threshold for text extraction (optional filtering).
        """
        self.min_length = min_length

    def parse(self, html: str) -> 'HtmlStructureManager':
        # Factory method to match existing interface
        return HtmlStructureManager(html)

class HtmlStructureManager:
    def __init__(self, html_content: str):
        """
        Initialize the manager.
        :param html_content: Original HTML string.
        """
        self.soup = BeautifulSoup(html_content, 'lxml') # Use lxml for robustness
        # Map to store node references for O(1) lookup
        self._node_map: Dict[str, Tag] = {}
        
        # Define tag strategies for extraction
        self.tag_strategies = {
            'title': ['title'],
            'heading': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
            'paragraph': ['p'],
            'list': ['ul', 'ol', 'dl'],
            'table': ['table'],
            # 'page_header': ['header'], 
            # 'page_footer': ['footer'],
            # 'semantic_container': ['article', 'section', 'main', 'aside', 'nav'] # Support finer granularity if needed
        }
        
        # Cache for parsed results
        self.extracted_elements: List[Dict] = []
        self._chunks: List[ContentChunk] = []
        self._parse_and_mark()

    def _generate_id(self) -> str:
        """Generate a unique short ID."""
        return f"geo-{str(uuid.uuid4())[:8]}"

    def _is_header_or_footer(self, tag: Tag, type_check: str) -> bool:
        """Helper to check if a tag is header/footer via semantic tag or class/id."""
        if tag.name == type_check: # e.g., <header>
            return True
        
        # Check class or id for keywords
        tag_classes = tag.get('class', []) or []
        # Ensure list
        if isinstance(tag_classes, str): 
            tag_classes = [tag_classes]
            
        tag_id = tag.get('id', '')
        
        identifiers = tag_classes + ([tag_id] if tag_id else [])
             
        for ident in identifiers:
            if isinstance(ident, str) and type_check in ident.lower():
                return True
        return False

    def _is_noise_match(self, ident: str, keyword: str) -> bool:
        """
        精确匹配噪声关键词，支持边界检测。

        匹配规则：
        - 完全匹配: sidebar == sidebar ✓
        - 前缀匹配: sidebar-left ✓ (keyword 在开头，后接分隔符)
        - 排除中间位置: content-sidebar-wrap ✗
        - 排除无边界: theiastickysidebar ✗
        - 排除布局修饰词: with-sidebar, has-sidebar ✗ (表示容器属性而非元素本身)
        """
        if ident == keyword:
            return True

        # 排除布局修饰前缀（表示"包含X"而非"是X"）
        layout_prefixes = ['with', 'has', 'no', 'without', 'content']

        # 只匹配 keyword 在开头的情况（前缀匹配）
        # 例如：sidebar-left, sidebar_nav
        prefix_pattern = rf'^{re.escape(keyword)}([-_]|$)'
        if re.search(prefix_pattern, ident):
            return True

        # 对于后缀位置（left-sidebar），检查前缀不是布局修饰词
        suffix_pattern = rf'(^|[-_]){re.escape(keyword)}$'
        if re.search(suffix_pattern, ident):
            # 提取 keyword 之前的部分
            idx = ident.rfind(keyword)
            if idx > 0:
                prefix_part = ident[:idx].rstrip('-_')
                # 如果前缀是布局修饰词，不匹配
                if prefix_part in layout_prefixes:
                    return False
            return True

        return False

    def _clean_dom(self):
        """
        Marks noise and interference elements in the DOM to be ignored during parsing.
        Does NOT remove them from the final HTML output (except checking scripts/styles).
        """
        # 1. Remove strict technical interference
        for tag in self.soup(['script', 'style', 'meta', 'noscript', 'svg']):
            tag.decompose()

        # 2. Mark Semantic & Heuristic Noise (Headers/Footers/Widgets)
        
        # A. Semantic tags to ignore (Mark instead of decompose)
        # 注意：检查内容量，避免误杀包含主内容的异常 HTML 结构
        for tag in self.soup(['nav', 'iframe']):
            # 如果 nav 包含大量文本或段落，可能是主内容误放
            text_len = len(tag.get_text(strip=True))
            p_count = len(tag.find_all('p'))
            if text_len > 5000 or p_count > 10:
                continue
            tag['data-geo-ignore'] = 'true'

        # 对于 header/footer，只在不包含 main/article 等主内容时才忽略
        for tag in self.soup(['header', 'footer']):
            # 如果 header/footer 包含 main 或 article，说明是异常的 HTML 结构，不应忽略
            if tag.find(['main', 'article', 'section']):
                continue
            # 新增：如果包含大量文本内容，可能是主内容误放在 header 中
            text_len = len(tag.get_text(strip=True))
            if text_len > 5000:  # 阈值：超过 5000 字符认为是主内容
                continue
            tag['data-geo-ignore'] = 'true'

        # B. Heuristic Noise Removal
        noise_candidates = self.soup.find_all(['div', 'section', 'aside', 'ul', 'ol'])
        
        noise_keywords = [
            # Header variations (handle "pageHeader", "siteHeader", etc.)
            'pageheader', 'page-header', 'siteheader', 'site-header', 'topheader', 'top-header', 'globalheader', 'global-header',
            # Footer variations (handle "pageFooter", "siteFooter", etc.)
            'pagefooter', 'page-footer', 'sitefooter', 'site-footer', 'bottomfooter', 'globalfooter', 
            # Other noise
            'sidebar', 'cookie', 'copyright', 'advertisement', 'banner-ads', 'widget-area'
        ]
        
        for tag in noise_candidates:
            # Skip if already removed
            if tag.parent is None: continue 
            
            # Safe-guard: Never remove main content
            # Check for common main content indicators
            tag_id = tag.get('id', '').lower()
            if (tag.get('role') == 'main' or
                tag_id in ['main', 'main-content', 'content', 'main-wrapper'] or
                tag.get('data-widget-def') == 'pageBody'):
                continue

            # Check Role (ARIA)
            role = tag.get('role')
            if role in ['banner', 'contentinfo', 'navigation', 'alert']:
                tag['data-geo-ignore'] = 'true'
                continue
                
            # Check Class and ID
            classes = tag.get('class', [])
            if isinstance(classes, str): classes = [classes]
            uid = tag.get('id', '')
            
            # Normalize identifiers for checking
            identifiers = [str(c).lower() for c in classes]
            if uid: identifiers.append(str(uid).lower())
            
            is_noise = False
            for ident in identifiers:
                # 1. Exact match for generic names
                if ident in ['header', 'footer', 'nav', 'menu', 'banner']:
                    is_noise = True
                    break
                # 2. Boundary-aware match for specific patterns
                # e.g., "sidebar-left" matches "sidebar" (at boundary)
                # but "with-sidebar" or "theiastickysidebar" do NOT match (keyword in middle/no boundary)
                if any(self._is_noise_match(ident, kw) for kw in noise_keywords):
                    is_noise = True
                    break

            if is_noise:
                # Additional safeguard: Don't ignore if contains main content elements
                # This prevents false positives like "penci_sidebar" that wraps main content
                if tag.find(['article', 'main']) or len(tag.find_all('p')) > 5:
                    continue
                tag['data-geo-ignore'] = 'true'

    def _parse_and_mark(self):
        """
        Core Step 1: Traverse DOM, identify target elements, assign unique IDs, and extract info.
        Update: Uses 'recursive' strategy to avoid duplicate extraction of nested elements (e.g., p inside table).
        """
        self._clean_dom()
        self.extracted_elements = []  # Reset
        self._node_map = {} # Reset
        

        # Priority Definition: We prefer capturing large structural blocks (Table, List) from the top level.
        # But for pure containers (div), we drill down to capture p.
        # Strategies are either leaf-oriented (p, h1-h6) or aggregate-oriented (table, list).
        # We use DFS. If we hit a high-level node (table, list), we mark it and do NOT drill down further into its children.
        
        target_tags = set()
        for tags in self.tag_strategies.values():
            target_tags.update(tags)

        # Traversal function
        def traverse(tag: Tag):
            # 0. Skip ignored elements
            if tag.get('data-geo-ignore') == 'true':
                return

            # 1. Attempt to match current node
            element_type = None
            
            # (A) Priority: Match structural blocks (Table, List)
            # If Table/List matched, treat as a whole to prevent fragmentation.
            if tag.name in self.tag_strategies['table']:
                element_type = 'table'
            elif tag.name in self.tag_strategies['list']:
                element_type = 'list'
            # (C) Basic Text Blocks
            elif tag.name in self.tag_strategies['heading']:
                element_type = 'heading'
            elif tag.name in self.tag_strategies['paragraph']:
                element_type = 'paragraph'
            
            # 2. Handle Match
            if element_type:
                # Filter logic: Skip empty p or very short content
                text = tag.get_text(strip=True)
                if len(text) > 0: # Can add min_length check here
                    # 复用已有 geo-id（如果存在且有效），否则生成新 ID
                    existing_id = tag.get('data-geo-id')
                    if existing_id and existing_id.startswith('geo-') and existing_id not in self._node_map:
                        geo_id = existing_id  # 复用已有 ID，且无冲突
                    else:
                        geo_id = self._generate_id()
                        tag['data-geo-id'] = geo_id
                    self._node_map[geo_id] = tag
                    
                    self.extracted_elements.append({
                        'id': geo_id,
                        'type': element_type,
                        'tag_name': tag.name,
                        'original_html': str(tag),
                        'text_content': text 
                    })
                    return # Stop recursing down this branch (Pruning)
            
            # 3. No match or non-target container (div, span, article, etc.), continue recursion.
            # Note: If element_type was 'page_header' and we returned, we won't catch h1 inside it.
            # Current Strategy: Table/List/Header/Footer are treated as single blocks.
            
            for child in tag.children:
                if isinstance(child, Tag):
                    traverse(child)

        if self.soup.body:
            traverse(self.soup.body)
        else:
            traverse(self.soup)

    # ---------------- API for Optimizer ----------------

    def get_clean_text(self) -> str:
        """Returns concatenated text for analysis (from initial parse snapshot)."""
        return "\n\n".join([e['text_content'] for e in self.extracted_elements])

    def get_current_text(self) -> str:
        """
        从当前 DOM 状态获取文本（非快照）。

        与 get_clean_text() 不同，此方法直接从当前 DOM 读取文本，
        能够反映所有已应用的修改。在 extracted_elements 可能过时的场景下使用。
        """
        if self.soup.body:
            return self.soup.body.get_text(separator="\n\n", strip=True)
        return self.soup.get_text(separator="\n\n", strip=True)

    def calculate_chunks(self, max_chunk_length: int = 2000, max_snippet_length: int = 10000) -> int:
        """
        Groups extracted elements into Chunks based on length and stores them internally.
        
        Args:
            max_chunk_length: Soft limit for each chunk size. Single elements larger than this are allowed as their own chunk.
            max_snippet_length: Global limit for the total text content processed. Elements beyond this limit are ignored.
        """
        self._chunks = []
        current_chunk_elements = []
        current_len = 0
        chunk_idx = 0
        total_used_len = 0

        for el in self.extracted_elements:
            text_len = len(el['text_content'])
            
            # Global Limit: Stop processing if total length exceeds max_snippet_length
            if total_used_len + text_len > max_snippet_length:
                # Ensure we have at least some content if the very first element is huge? 
                # For now, strictly respect the limit (unless nothing has been added, which might be an edge case)
                if total_used_len > 0: 
                    break
                # If total_used_len == 0 (first element is huge), we proceed to include it 
                # to avoid empty result, assuming user wants at least something.

            total_used_len += text_len
            
            # Chunking Logic:
            # If adding this element makes the current chunk too big...
            if current_len + text_len > max_chunk_length and current_chunk_elements:
                # ...finish the current chunk
                self._chunks.append(ContentChunk(chunk_idx, current_chunk_elements))
                chunk_idx += 1
                current_chunk_elements = []
                current_len = 0
            
            # Add element to current chunk (blocks if merging, or takes strictly if empty)
            current_chunk_elements.append(el)
            current_len += text_len
            
        if current_chunk_elements:
             self._chunks.append(ContentChunk(chunk_idx, current_chunk_elements))
             
    
    def get_chunk_tool_args(self, index: Optional[int]) -> Dict[str, str]:
        """
        Returns context args for tool execution: target_content, context_before, context_after.
        """
        if not self._chunks:
            return {"target_content": "", "context_before": "", "context_after": ""}
        
        # Clamp index
        idx = 0 if index is None else max(0, min(index, len(self._chunks) - 1))
        
        target_chunk = self._chunks[idx]
        prev_chunk = self._chunks[idx-1] if idx > 0 else None
        next_chunk = self._chunks[idx+1] if idx < len(self._chunks) - 1 else None
        
        return {
            "target_content": target_chunk.html,
            "context_before": prev_chunk.text if prev_chunk else "",
            "context_after": next_chunk.text if next_chunk else ""
        }

    def replace_chunk_by_index(self, index: Optional[int], new_html: str, highlight: bool = True) -> bool:
        """
        Replaces the chunk at the specified index with new HTML.
        """
        if not self._chunks:
            return False
            
        idx = 0 if index is None else max(0, min(index, len(self._chunks) - 1))
        target_chunk = self._chunks[idx]
        
        return self.replace_chunk(target_chunk, new_html, highlight)

    def replace_chunk(self, chunk: ContentChunk, new_html: str, highlight: bool = True) -> bool:
        """
        Replaces a Chunk (which may contain multiple elements) with new HTML.
        
        Strategy:
        1. Parse new HTML.
        2. Replace the FIRST element of the chunk with the NEW content.
        3. Remove/Highlight the REST of the elements in the chunk.
        """
        if not chunk._elements_data:
            return False

        try:
            # 1. Prepare New Content (使用 lxml 保持与初始解析一致)
            new_soup = BeautifulSoup(new_html, 'lxml')
            # Extract actual elements from body if present
            if new_soup.body:
                 new_tags = list(new_soup.body.contents)
            else:
                 new_tags = list(new_soup.contents)
            
            # 2. Handle First Element (Anchor)
            first_el_data = chunk._elements_data[0]
            first_tag_ref = self._node_map.get(first_el_data['id'])
            
            if not first_tag_ref:
                logger.error(f"Anchor tag {first_el_data['id']} lost.")
                return False

            # Optional Highlight Container
            if highlight:
                container = self.soup.new_tag("div", attrs={
                    "class": "geo-modified-chunk",
                    "style": "border-left: 4px solid #4CAF50; background-color: #f1f8e9; padding: 10px; margin: 10px 0;"
                })
                for t in new_tags:
                    container.append(t) # Moves t into container
                
                # Replace first tag with Container
                first_tag_ref.replace_with(container)
            else:
                # Replace first tag with all new tags. 
                # replace_with supports inserting multiple elements? No, usually one.
                # Use insert_after/before loop or replace with first and insert rest.
                first_tag_ref.clear() # Clear content
                # This part is tricky if new_tags > 1. 
                # Simplest: Insert new tags before first_tag_ref, then decompose first_tag_ref.
                for t in new_tags:
                    first_tag_ref.insert_before(t)
                first_tag_ref.decompose()

            # 3. Remove Remaining Elements of the Chunk
            # Since we injected ALL new content at the position of the first element,
            # the subsequent elements in the chunk (which represented the old text) are now obsolete.
            for el_data in chunk._elements_data[1:]:
                tag_ref = self._node_map.get(el_data['id'])
                if tag_ref:
                    tag_ref.decompose()
            
            return True

        except Exception as e:
            logger.error(f"Chunk replacement failed: {e}")
            return False

    def export_html(self, clean_internal_attributes: bool = True) -> str:
        """
        Export the current HTML state.
        
        Args:
            clean_internal_attributes: If True, removes internal attributes used for processing (e.g. data-geo-ignore).
        """
        if clean_internal_attributes:
            # Remove data-geo-ignore markers so they don't appear in the final output
            for tag in self.soup.find_all(attrs={"data-geo-ignore": True}):
                del tag['data-geo-ignore']

        return str(self.soup)
    
    def format_indexed_content(self):
        return "\n\n".join(
            [f">> [CHUNK_ID: {i}]\n{chunk_text.text}" for i, chunk_text in enumerate(self._chunks)]
        )

    # ---------------- 双结构策略 API（用于 Batch GEO Agent）----------------

    def format_indexed_content_with_live_dom(self, live_structure: 'HtmlStructureManager') -> str:
        """
        使用冻结索引 + 活跃内容生成 indexed_content。

        Args:
            live_structure: 活跃结构（包含最新修改的 DOM）

        Returns:
            str: 带索引的内容，索引来自 self（冻结结构），文本内容来自 live_structure

        工作原理：
        - self（冻结结构）的 _chunks 提供稳定的索引
        - 通过 geo-id 在 live_structure 中查找对应元素获取最新文本
        - 如果 geo-id 在 live_structure 中不存在，使用冻结结构的原始文本
        """
        if not self._chunks:
            return ""

        result_parts = []

        for chunk_idx, chunk in enumerate(self._chunks):
            chunk_texts = []

            for el_data in chunk._elements_data:
                geo_id = el_data.get('id')

                if geo_id and geo_id in live_structure._node_map:
                    # 从活跃结构获取最新文本
                    live_tag = live_structure._node_map[geo_id]
                    text = live_tag.get_text(strip=True)
                else:
                    # 回退到冻结结构的原始文本
                    text = el_data.get('text_content', '')

                if text:
                    chunk_texts.append(text)

            combined_text = "\n\n".join(chunk_texts)
            result_parts.append(f">> [CHUNK_ID: {chunk_idx}]\n{combined_text}")

        return "\n\n".join(result_parts)

    def get_chunk_tool_args_from_live(
        self, live_structure: 'HtmlStructureManager', frozen_index: int
    ) -> Dict[str, str]:
        """
        使用冻结索引定位，从活跃结构提取工具参数。

        Args:
            live_structure: 活跃结构（包含最新修改的 DOM）
            frozen_index: 冻结结构中的 chunk 索引

        Returns:
            Dict: 包含 target_content, context_before, context_after

        工作原理：
        - 使用 self（冻结结构）的 _chunks 确定哪些 geo-id 属于目标 chunk
        - 从 live_structure 的 _node_map 中提取这些元素的最新 HTML/文本
        - 上下文也从 live_structure 获取（如果 geo-id 存在）
        """
        if not self._chunks:
            return {"target_content": "", "context_before": "", "context_after": ""}

        # Clamp index
        idx = max(0, min(frozen_index, len(self._chunks) - 1))

        target_chunk = self._chunks[idx]
        prev_chunk = self._chunks[idx - 1] if idx > 0 else None
        next_chunk = self._chunks[idx + 1] if idx < len(self._chunks) - 1 else None

        def get_chunk_html_from_live(chunk: ContentChunk) -> str:
            """从活跃结构获取 chunk 的 HTML"""
            html_parts = []
            for el_data in chunk._elements_data:
                geo_id = el_data.get('id')
                if geo_id and geo_id in live_structure._node_map:
                    html_parts.append(str(live_structure._node_map[geo_id]))
                else:
                    # 回退到原始 HTML
                    html_parts.append(el_data.get('original_html', ''))
            return "\n".join(html_parts)

        def get_chunk_text_from_live(chunk: ContentChunk) -> str:
            """从活跃结构获取 chunk 的文本"""
            text_parts = []
            for el_data in chunk._elements_data:
                geo_id = el_data.get('id')
                if geo_id and geo_id in live_structure._node_map:
                    text_parts.append(live_structure._node_map[geo_id].get_text(strip=True))
                else:
                    text_parts.append(el_data.get('text_content', ''))
            return "\n\n".join(text_parts)

        return {
            "target_content": get_chunk_html_from_live(target_chunk),
            "context_before": get_chunk_text_from_live(prev_chunk) if prev_chunk else "",
            "context_after": get_chunk_text_from_live(next_chunk) if next_chunk else ""
        }

    def apply_modification_to_live(
        self, live_structure: 'HtmlStructureManager', frozen_index: int, new_html: str, highlight: bool = False
    ) -> bool:
        """
        在活跃结构上应用修改，位置由冻结索引确定。

        Args:
            live_structure: 活跃结构（将被修改）
            frozen_index: 冻结结构中的 chunk 索引
            new_html: 新的 HTML 内容
            highlight: 是否高亮修改区域

        Returns:
            bool: 是否成功应用修改

        工作原理：
        - 使用 self（冻结结构）的 _chunks 确定哪些 geo-id 需要被替换
        - 在 live_structure 中查找这些 geo-id 对应的元素
        - 用 new_html 替换第一个元素，删除其余元素
        """
        if not self._chunks:
            logger.warning("No chunks in frozen structure")
            return False

        # Clamp index
        idx = max(0, min(frozen_index, len(self._chunks) - 1))
        target_chunk = self._chunks[idx]

        if not target_chunk._elements_data:
            logger.warning(f"Chunk {idx} has no elements")
            return False

        try:
            # 1. 准备新内容（使用 lxml 保持与初始解析一致）
            new_soup = BeautifulSoup(new_html, 'lxml')
            if new_soup.body:
                new_tags = list(new_soup.body.contents)
            else:
                new_tags = list(new_soup.contents)

            # 2. 查找第一个元素（锚点）在活跃结构中的位置
            first_el_data = target_chunk._elements_data[0]
            first_geo_id = first_el_data.get('id')

            if not first_geo_id:
                logger.error("First element has no geo-id")
                return False

            first_tag_ref = live_structure._node_map.get(first_geo_id)
            if not first_tag_ref or first_tag_ref.parent is None:
                logger.error(f"Anchor tag {first_geo_id} not found in live structure or already removed")
                return False

            # 3. 替换第一个元素
            if highlight:
                container = live_structure.soup.new_tag("div", attrs={
                    "class": "geo-modified-chunk",
                    "style": "border-left: 4px solid #4CAF50; background-color: #f1f8e9; padding: 10px; margin: 10px 0;"
                })
                for t in new_tags:
                    container.append(t)
                first_tag_ref.replace_with(container)
            else:
                # 插入新内容，然后删除原元素
                for t in new_tags:
                    first_tag_ref.insert_before(t)
                first_tag_ref.decompose()

            # 4. 删除 chunk 中其余元素（在活跃结构中）
            for el_data in target_chunk._elements_data[1:]:
                geo_id = el_data.get('id')
                if geo_id and geo_id in live_structure._node_map:
                    tag_ref = live_structure._node_map[geo_id]
                    if tag_ref and tag_ref.parent is not None:
                        tag_ref.decompose()

            return True

        except Exception as e:
            logger.error(f"apply_modification_to_live failed: {e}")
            return False
