
from bs4 import BeautifulSoup

class HTMLFragmentProcessor:
    """
    Helper class to inspect and modify HTML fragments using BeautifulSoup.
    Allows tools to operate on specific parts of the DOM without rewriting the whole string.
    """
    def __init__(self, html_content: str):
        # Use a wrapper div to ensure we can handle multiple top-level elements if necessary
        # but for fragments, usually parser takes it okay. 
        self.soup = BeautifulSoup(html_content, 'html.parser')

    def get_text_content(self) -> str:
        """Returns clean text content for LLM context."""
        return self.soup.get_text(separator=" ", strip=True)

    def prepend_container(self, content: str, tag='div', **attrs):
        """Prepends a new container with the given content at the top."""
        new_tag = self.soup.new_tag(tag, **attrs)
        
        # Parse content as HTML chunk
        inner_soup = BeautifulSoup(content, 'html.parser')
        
        # If inner soup has body (full doc parsed), take body children, else take all
        if inner_soup.body:
             children = list(inner_soup.body.children)
        else:
             children = list(inner_soup.children)
             
        for child in children:
            new_tag.append(child)
        
        self.soup.insert(0, new_tag)

    def to_html(self) -> str:
        """Returns the modified HTML string."""
        return str(self.soup)

def build_context_section(context_before: str, context_after: str) -> str:
    """Helper to build the read-only context string."""
    section = ""
    if context_before or context_after:
        section = "\n📖 SURROUNDING CONTEXT (READ-ONLY - for flow, DO NOT include in output):\n"
        if context_before:
            section += f"\n--- CONTENT BEFORE (READ-ONLY) ---\n{context_before}\n--- END CONTENT BEFORE ---\n"
        if context_after:
            section += f"\n--- CONTENT AFTER (READ-ONLY) ---\n{context_after}\n--- END CONTENT AFTER ---\n"
    return section

# Standard preservation prompt fragment
PRESERVATION_RULES = """
⚠️ CRITICAL PRESERVATION RULES:
1. **CORE FOCUS**: The document MUST remain focused on the core idea - DO NOT drift to other topics. NEVER change what the document is fundamentally about.
2. **NO INFORMATION LOSS**: You are forbidden from deleting existing facts, numbers, or entities. You may rephrase sentences to smoothly incorporate new facts, but DO NOT delete original content.
3. **CONTEXT INTEGRITY**: Do not contradict the "CONTENT BEFORE" or "CONTENT AFTER". Weave the new information naturally into relevant sections where it fits contextually.
4. **FORMAT & STYLE**: Maintain the same format (HTML or text) and writing style as the input.
5. **PRESERVE MODIFICATIONS**: PRESERVE all previous modifications - DO NOT undo or contradict them.
6. **OUTPUT ISOLATION**: ONLY output the modified TARGET CONTENT section - DO NOT include the context sections.
"""