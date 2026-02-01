import logging
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import html2text
from markdownify import markdownify as md
from geo_agent.config import load_config
import trafilatura

# Abstract Base Class
class BaseParser(ABC):
    @abstractmethod
    def parse(self, html: str) -> str:
        pass

class MarkdownParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        使用 markdownify 将 HTML 转换为 Markdown。
        这种方式保留了超链接和图片引用，适合 LLM 阅读。
        """
        try:
            # heading_style="ATX" 表示使用 # 标题
            return md(html, heading_style="ATX", strip=['script', 'style'])
        except Exception as e:
            logging.error(f"Markdown 转换失败: {e}")
            return ""

class Html2TextParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        使用 html2text 库。它生成的 Markdown 风格文本非常干净，
        是 LangChain HTML2Text transformer 的核心。
        """
        try:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_tables = False
            h.body_width = 0  # 不自动换行
            return h.handle(html)
        except Exception as e:
            logging.error(f"Html2Text 转换失败: {e}")
            return ""

class Bs4Parser(BaseParser):
    def parse(self, html: str) -> str:
        """
        使用 BeautifulSoup 提取纯文本。
        去除所有标签，只留文字。适合做向量检索 (Embedding)。
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
            # 移除脚本和样式
            for script in soup(["script", "style", "head", "meta", "noscript"]):
                script.decompose()
            
            text = soup.get_text(separator='\n\n')
            # 清理多余空行
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            logging.error(f"BS4 提取文本失败: {e}")
            return ""

class NewspaperParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        使用 newspaper3k 提取文本。
        """
        try:
            from newspaper import Article

            article = Article('')
            article.set_html(html)
            article.parse()
            return article.text
        except ImportError:
            logging.warning("未安装 'newspaper3k' 库，跳过该解析方法。")
            return ""
        except Exception as e:
            logging.error(f"Newspaper3k 转换失败: {e}")
            return ""

class UnstructuredParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        尝试使用 unstructured 库进行分区解析 (Partition)。
        如果未安装则降级处理。
        """
        try:
            from unstructured.partition.html import partition_html
            elements = partition_html(text=html)
            # 将提取出的元素重新组合成文本
            return "\n\n".join([str(el) for el in elements])
        except ImportError:
            logging.warning("未安装 'unstructured' 库，跳过该解析方法。")
            return ""
        except Exception as e:
            logging.error(f"Unstructured 解析失败: {e}")
            return ""

class TrafilaturaParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        使用 trafilatura 提取主要内容。
        需要安装 trafilatura 库。
        """
        try:
            downloaded = trafilatura.extract(html)
            return downloaded if downloaded else ""
        except ImportError:
            logging.warning("未安装 'trafilatura' 库，跳过该解析方法。")
            return ""
        except Exception as e:
            logging.error(f"Trafilatura 提取失败: {e}")
            return ""

class ReadabilityParser(BaseParser):
    def parse(self, html: str) -> str:
        """
        使用 readability-lxml 提取主要内容。
        需要安装 readability-lxml 库。
        """
        try:
            from readability import Document
            doc = Document(html)
            return doc.summary()
        except ImportError:
            logging.warning("未安装 'readability-lxml' 库，跳过该解析方法。")
            return ""
        except Exception as e:
            logging.error(f"Readability 提取失败: {e}")
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

        # 配置的方法不可用/提取失败时，降级到 Bs4Parser，避免返回空文本导致下游不可用。
        return Bs4Parser().parse(html)

    def to_clean_text_bs4(self, html: str) -> str:
        """
        兼容旧接口：强制使用 BS4 提取纯文本。
        """
        return Bs4Parser().parse(html)
