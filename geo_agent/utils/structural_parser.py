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
        Precise noise keyword matching with boundary detection.

        Matching rules:
        - Exact match: sidebar == sidebar ✓
        - Prefix match: sidebar-left ✓ (keyword at start, followed by separator)
        - Exclude middle position: content-sidebar-wrap ✗
        - Exclude no boundary: theiastickysidebar ✗
        - Exclude layout modifiers: with-sidebar, has-sidebar ✗ (indicates container property, not element itself)
        """
        if ident == keyword:
            return True

        # Exclude layout modifier prefixes (indicates "contains X" rather than "is X")
        layout_prefixes = ['with', 'has', 'no', 'without', 'content']

        # Only match when keyword is at the beginning (prefix match)
        # Examples: sidebar-left, sidebar_nav
        prefix_pattern = rf'^{re.escape(keyword)}([-_]|$)'
        if re.search(prefix_pattern, ident):
            return True

        # For suffix position (left-sidebar), check prefix is not a layout modifier
        suffix_pattern = rf'(^|[-_]){re.escape(keyword)}$'
        if re.search(suffix_pattern, ident):
            # Extract the part before keyword
            idx = ident.rfind(keyword)
            if idx > 0:
                prefix_part = ident[:idx].rstrip('-_')
                # If prefix is a layout modifier, don't match
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
        # Note: Check content volume to avoid false positives on abnormal HTML structures containing main content
        for tag in self.soup(['nav', 'iframe']):
            # If nav contains large amounts of text or paragraphs, main content may be misplaced
            text_len = len(tag.get_text(strip=True))
            p_count = len(tag.find_all('p'))
            if text_len > 5000 or p_count > 10:
                continue
            tag['data-geo-ignore'] = 'true'

        # For header/footer, only ignore when not containing main/article and other main content
        for tag in self.soup(['header', 'footer']):
            # If header/footer contains main or article, it's an abnormal HTML structure, should not ignore
            if tag.find(['main', 'article', 'section']):
                continue
            # Additional: if contains large text content, main content may be misplaced in header
            text_len = len(tag.get_text(strip=True))
            if text_len > 5000:  # Threshold: over 5000 characters considered main content
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
                    # Reuse existing geo-id (if exists and valid), otherwise generate new ID
                    existing_id = tag.get('data-geo-id')
                    if existing_id and existing_id.startswith('geo-') and existing_id not in self._node_map:
                        geo_id = existing_id  # Reuse existing ID, no conflict
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
        Get text from current DOM state (not snapshot).

        Unlike get_clean_text(), this method reads text directly from current DOM,
        reflecting all applied modifications. Use when extracted_elements may be outdated.
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
            # 1. Prepare New Content (use lxml to maintain consistency with initial parsing)
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

    # ---------------- Dual Structure Strategy API (for Batch GEO Agent) ----------------

    def format_indexed_content_with_live_dom(self, live_structure: 'HtmlStructureManager') -> str:
        """
        Generate indexed_content using frozen index + live content.

        Args:
            live_structure: Active structure (DOM with latest modifications)

        Returns:
            str: Indexed content, indexes from self (frozen structure), text content from live_structure

        How it works:
        - self (frozen structure)'s _chunks provide stable indexes
        - Lookup corresponding elements in live_structure via geo-id to get latest text
        - If geo-id doesn't exist in live_structure, use original text from frozen structure
        """
        if not self._chunks:
            return ""

        result_parts = []

        for chunk_idx, chunk in enumerate(self._chunks):
            chunk_texts = []

            for el_data in chunk._elements_data:
                geo_id = el_data.get('id')

                if geo_id and geo_id in live_structure._node_map:
                    # Get latest text from live structure
                    live_tag = live_structure._node_map[geo_id]
                    text = live_tag.get_text(strip=True)
                else:
                    # Fallback to original text from frozen structure
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
        Use frozen index for positioning, extract tool arguments from live structure.

        Args:
            live_structure: Live structure (DOM with latest modifications)
            frozen_index: Chunk index in frozen structure

        Returns:
            Dict: Contains target_content, context_before, context_after

        How it works:
        - Use self (frozen structure)'s _chunks to determine which geo-ids belong to target chunk
        - Extract latest HTML/text for these elements from live_structure's _node_map
        - Context is also obtained from live_structure (if geo-id exists)
        """
        if not self._chunks:
            return {"target_content": "", "context_before": "", "context_after": ""}

        # Clamp index
        idx = max(0, min(frozen_index, len(self._chunks) - 1))

        target_chunk = self._chunks[idx]
        prev_chunk = self._chunks[idx - 1] if idx > 0 else None
        next_chunk = self._chunks[idx + 1] if idx < len(self._chunks) - 1 else None

        def get_chunk_html_from_live(chunk: ContentChunk) -> str:
            """Get chunk HTML from live structure"""
            html_parts = []
            for el_data in chunk._elements_data:
                geo_id = el_data.get('id')
                if geo_id and geo_id in live_structure._node_map:
                    html_parts.append(str(live_structure._node_map[geo_id]))
                else:
                    # Fallback to original HTML
                    html_parts.append(el_data.get('original_html', ''))
            return "\n".join(html_parts)

        def get_chunk_text_from_live(chunk: ContentChunk) -> str:
            """Get chunk text from live structure"""
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
        Apply modification to live structure, position determined by frozen index.

        Args:
            live_structure: Live structure (will be modified)
            frozen_index: Chunk index in frozen structure
            new_html: New HTML content
            highlight: Whether to highlight modified area

        Returns:
            bool: Whether modification was successfully applied

        How it works:
        - Use self (frozen structure)'s _chunks to determine which geo-ids need to be replaced
        - Find the elements corresponding to these geo-ids in live_structure
        - Replace first element with new_html, delete remaining elements
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
            # 1. Prepare new content (use lxml to maintain consistency with initial parsing)
            new_soup = BeautifulSoup(new_html, 'lxml')
            if new_soup.body:
                new_tags = list(new_soup.body.contents)
            else:
                new_tags = list(new_soup.contents)

            # 2. Find first element (anchor) position in live structure
            first_el_data = target_chunk._elements_data[0]
            first_geo_id = first_el_data.get('id')

            if not first_geo_id:
                logger.error("First element has no geo-id")
                return False

            first_tag_ref = live_structure._node_map.get(first_geo_id)
            if not first_tag_ref or first_tag_ref.parent is None:
                logger.error(f"Anchor tag {first_geo_id} not found in live structure or already removed")
                return False

            # 3. Replace first element
            if highlight:
                container = live_structure.soup.new_tag("div", attrs={
                    "class": "geo-modified-chunk",
                    "style": "border-left: 4px solid #4CAF50; background-color: #f1f8e9; padding: 10px; margin: 10px 0;"
                })
                for t in new_tags:
                    container.append(t)
                first_tag_ref.replace_with(container)
            else:
                # Insert new content, then delete original element
                for t in new_tags:
                    first_tag_ref.insert_before(t)
                first_tag_ref.decompose()

            # 4. Delete remaining elements in chunk (in live structure)
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
