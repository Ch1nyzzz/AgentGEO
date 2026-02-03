import logging
from geo_agent.config import load_config
from .browser import HtmlBrowser
from geo_agent.utils import HtmlParser, DataSaver
from geo_agent.core import SearchResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class ContentLoader:
    """
    Responsible for downloading and parsing content from URLs.
    Handles PDF, TXT, and HTML.
    """
    def __init__(self, config_path='geo_agent/config.yaml'):
        self.config = load_config(config_path)
        self.browser = HtmlBrowser(config_path)
        self.parser = HtmlParser()
        # Use the same cache directory as HtmlBrowser
        data_config = self.config.get('data', {})
        cache_dir = data_config.get('html_db_path', 'outputs')
        self.saver = DataSaver(base_dir=cache_dir)
        self._cache = {}

    def process(self, result: SearchResult):
        url = result.url
        if url in self._cache:
            return self._cache[url]

        # Use UUID as filename if available, otherwise hash the URL
        if hasattr(result, 'uuid') and result.uuid:
            file_alias = result.uuid
        else:
            file_alias = self.saver.clean_filename(url)

        # First try to load from parsed_content cache
        for suffix in ['trafilatura_parsed', 'readability_parsed', 'parsed']:
            cached_content = self.saver.load(f"parsed_content/{file_alias}_{suffix}", ext="txt")
            if cached_content:
                self._cache[url] = cached_content
                return cached_content

        content = None
        method_suffix = "parsed"

        url_lower = url.lower()

        if url_lower.endswith('.pdf'):
            content = self.browser.browser(result)
            method_suffix = "pdf_parsed"

        elif url_lower.endswith('.txt'):
            content = self.browser.fetch_txt_text(url)
            method_suffix = "txt_parsed"

        else:
            best_html = self.browser.browser(result)
            if best_html:
                content = self.parser.parse(best_html)
                method = getattr(self.parser, 'method', 'parsed')
                method_suffix = f"{method}_parsed"

        if content:
            self._cache[url] = content
            self.saver.save(content, f"parsed_content/{file_alias}_{method_suffix}", "txt")
            return content
        else:
            return None
