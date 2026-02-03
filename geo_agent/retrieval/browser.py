import os
import logging
from typing import Optional
import requests
import httpx
from geo_agent.config import load_config
from geo_agent.search_engine.chatnoir import ChatNoirClient
from geo_agent.utils.storage import DataSaver
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class HtmlBrowser:
    """Responsible for fetching HTML source code, supporting caching."""
    
    def __init__(self, config_path='geo_agent/config.yaml'):
        self.config = load_config(config_path)
        self.data_config = self.config.get('data', {})
        self.mode = self.data_config.get('mode', 'cw22')      
        self.db_path = self.data_config.get('html_db_path', 'cw22_search_dataset/data/html_store')

        self.fetch_config = self.config.get('html_browser', {})
        self.method = self.fetch_config.get('method', 'requests')
        self.saver = DataSaver(base_dir=self.db_path)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }

    def browser(self, result) -> Optional[str]:
        """
        Fetch HTML content.
        If mode is 'cw22', try local store then ChatNoir.
        Otherwise, check local cache then fetch online.
        """
        url = result.url

        if self.mode == 'cw22':
            html_body = self.fetch_cw22(result.uuid)
            if html_body:
                return html_body
            else:
                client = ChatNoirClient()
                html_body = client.get_html_content(result.uuid)
                if html_body:
                    self.saver.save(html_body, result.uuid)
                return html_body
        
        # Common logic for online/offline checking cache first
        if hasattr(result, 'uuid') and result.uuid:
            file_alias = result.uuid
        else:
            file_alias = self.saver.clean_filename(url)
        
        cached_content = self.saver.load(file_alias)
        if cached_content:
            # logging.info(f"Read from local cache")
            return cached_content
            
        content = None
        if url.lower().endswith(".pdf"):
            content = self.fetch_pdf_text(url)
        elif url.lower().endswith(".txt"):
            content = self.fetch_txt_text(url)
        else:
            if self.method == "requests":
                content = self._fetch_requests(url)
            elif self.method == "httpx":
                content = self._fetch_httpx(url)
            elif self.method == "playwright":
                content = self._fetch_playwright(url)
            else:
                content = self._fetch_requests(url)
                
            if content:
                self.saver.save(content, file_alias, ext="html")
            
        return content

    def _fetch_requests(self, url: str) -> Optional[str]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=15, verify=False)
            resp.encoding = resp.apparent_encoding
            return resp.text
        except Exception as e:
            logging.error(f"[Requests] Error: {e}")
            return None

    def _fetch_httpx(self, url: str) -> Optional[str]:
        try:
            with httpx.Client(headers=self.headers, http2=True, timeout=15, verify=False) as client:
                resp = client.get(url)
                return resp.text
        except Exception as e:
            logging.error(f"[Httpx] Error: {e}")
            return None

    def _fetch_playwright(self, url: str) -> Optional[str]:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=self.headers['User-Agent'])
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                content = page.content()
                browser.close()
                return content
        except Exception as e:
            logging.error(f"[Playwright] Error: {e}")
            return None

    def fetch_pdf_text(self, url: str) -> Optional[str]:
        """
        Download PDF and extract text content.
        Uses pdfplumber library for text extraction.
        """
        try:
            import pdfplumber
            resp = requests.get(url, headers=self.headers, timeout=15, verify=False)
            resp.raise_for_status()
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip() if text else None
        except ImportError:
            logging.error("'pdfplumber' library not installed, cannot process PDF. Please run: pip install pdfplumber")
            return None
        except Exception as e:
            logging.error(f"PDF extraction failed: {e}")
            return None

    def fetch_txt_text(self, url: str) -> Optional[str]:
        """
        Download TXT file and return its content.
        """
        try:
            resp = requests.get(url, headers=self.headers, timeout=15, verify=False)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logging.error(f"TXT read failed: {e}")
            return None

    def fetch_cw22(self, uuid: str) -> Optional[str]:
        """
        Fetch HTML content from ClueWeb22 local storage.
        """
        if not self.db_path or not os.path.exists(self.db_path):
            logging.error("ClueWeb22 database path invalid or does not exist.")
            return None

        # uuid as filename, ext="html" will automatically add .html suffix
        return self.saver.load(uuid, ext="html")