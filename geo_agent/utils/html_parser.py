import logging
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import html2text
from markdownify import markdownify as md
from geo_agent.config import load_config

# Abstract Base Class
class BaseParser(ABC):
    @abstractmethod
    def parse(self, html: str) -> str:
        pass

class MarkdownParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        Convert HTML to Markdown using markdownify.
        This method preserves hyperlinks and image references, suitable for LLM reading.
        """
        try:
            # heading_style="ATX" means using # for headings
            return md(html, heading_style="ATX", strip=['script', 'style'])
        except Exception as e:
            logging.error(f"Markdown conversion failed: {e}")
            return ""

class Html2TextParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        Use html2text library. It generates very clean Markdown-style text,
        which is the core of LangChain HTML2Text transformer.
        """
        try:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_tables = False
            h.body_width = 0  # Disable automatic line wrapping
            return h.handle(html)
        except Exception as e:
            logging.error(f"Html2Text conversion failed: {e}")
            return ""

class Bs4Parser(BaseParser):
    def parse(self, html: str) -> str:
        """
        Extract plain text using BeautifulSoup.
        Removes all tags, leaving only text. Suitable for vector retrieval (Embedding).
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
            # Remove scripts and styles
            for script in soup(["script", "style", "head", "meta", "noscript"]):
                script.decompose()
            
            text = soup.get_text(separator='\n\n')
            # Clean up extra blank lines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            logging.error(f"BS4 text extraction failed: {e}")
            return ""

class NewspaperParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        Extract text using newspaper3k.
        """
        try:
            from newspaper import Article

            article = Article('')
            article.set_html(html)
            article.parse()
            return article.text
        except ImportError:
            logging.warning("'newspaper3k' library not installed, skipping this parsing method.")
            return ""
        except Exception as e:
            logging.error(f"Newspaper3k conversion failed: {e}")
            return ""

class UnstructuredParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        Try to use unstructured library for partition parsing.
        Falls back gracefully if not installed.
        """
        try:
            from unstructured.partition.html import partition_html
            elements = partition_html(text=html)
            # Recombine extracted elements into text
            return "\n\n".join([str(el) for el in elements])
        except ImportError:
            logging.warning("'unstructured' library not installed, skipping this parsing method.")
            return ""
        except Exception as e:
            logging.error(f"Unstructured parsing failed: {e}")
            return ""

class TrafilaturaParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        Extract main content using trafilatura.
        Requires trafilatura library to be installed.
        """
        try:
            import trafilatura
            downloaded = trafilatura.extract(html)
            return downloaded if downloaded else ""
        except ImportError:
            logging.warning("'trafilatura' library not installed, skipping this parsing method.")
            return ""
        except Exception as e:
            logging.error(f"Trafilatura extraction failed: {e}")
            return ""

class ReadabilityParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        Extract main content using readability-lxml.
        Requires readability-lxml library to be installed.
        """
        try:
            from readability import Document
            doc = Document(html)
            return doc.summary()
        except ImportError:
            logging.warning("'readability-lxml' library not installed, skipping this parsing method.")
            return ""
        except Exception as e:
            logging.error(f"Readability extraction failed: {e}")
            return ""

class HtmlParser:
    """
    Facade class to maintain backward compatibility and provide easy access.
    """
    def __init__(self, config_path='geo_agent/config.yaml'):
        try:
            self.config = load_config(config_path)
        except FileNotFoundError:
            logging.warning(f"Config file not found at {config_path}, using default settings.")
            self.config = {}
            
        self.method = self.config.get('html_parser', {}).get('method', 'trafilatura')
        self.parser_strategy = self._get_parser_strategy(self.method)

    def _get_parser_strategy(self, method: str) -> BaseParser:
        if method == 'markdown':
            return MarkdownParser()
        elif method == 'html2text':
            return Html2TextParser()
        elif method == 'newspaper':
            return NewspaperParser()
        elif method == 'unstructured':
            return UnstructuredParser()
        elif method == 'trafilatura':
            return TrafilaturaParser()
        elif method == 'readability':
            return ReadabilityParser()
        else:
            return Bs4Parser()

    def parse(self, html: str) -> str:
        result = self.parser_strategy.parse(html)
        if result:
            return result

        # Fall back to Bs4Parser when configured method is unavailable or extraction fails, to avoid returning empty text.
        return Bs4Parser().parse(html)

    def to_clean_text_bs4(self, html: str) -> str:
        """
        Legacy interface compatibility: Force use BS4 to extract plain text.
        """
        return Bs4Parser().parse(html)
