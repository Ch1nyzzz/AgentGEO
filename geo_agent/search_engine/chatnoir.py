from geo_agent.core.models import SearchResult
import logging
import requests
from requests.adapters import HTTPAdapter
from geo_agent.config import API_KEY, API_BASE_URL, INDEX_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class ChatNoirClient:
    def __init__(self, pool_size: int = 20, max_results: int = 14):
        self.session = requests.Session()
        self.max_results = max_results
        # Increase connection pool size to avoid "Connection pool is full" warning
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def search(self, query: str, size: int = None):
        """
        Execute search
        API: POST /api/v1/_search
        """
        if size is None:
            size = self.max_results
        url = f"{API_BASE_URL}/api/v1/_search"
        payload = {
            "query": query,
            "index": [INDEX_NAME],
            "size": size,
            "pretty": False,
            "explain": False
        }

        # Add retry mechanism
        max_retries = 3
        retry_delay = 2
        
        for retry in range(max_retries):
            try:
                # Disable SSL verification to avoid SSL errors
                self.session.verify = False
                response = self.session.post(url, json=payload, timeout=30)
                json_resp = response.json()

                if response.status_code != 200:
                    logger.error(f"Search API Error {response.status_code}: {response.text}")
                    if retry < max_retries - 1:
                        logger.info(f"Retrying search for query '{query}' ({retry+1}/{max_retries})...")
                        import time
                        time.sleep(retry_delay)
                        continue
                    return []
                
                # ChatNoir returns results in the 'results' list
                return [SearchResult(idx=i,
                                 title=result.get('title').replace("<em>", "").replace("</em>", ""),
                                 snippet=result.get('snippet').replace("<em>", "").replace("</em>", ""),
                                 url=result.get('target_uri'),
                                 uuid=result.get('uuid')) for i, result in enumerate(json_resp.get('results', []))]

            except requests.exceptions.SSLError as e:
                logger.error(f"SSL Error for search query '{query}': {e}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying search with SSL disabled for query '{query}' ({retry+1}/{max_retries})...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return []
            except ConnectionResetError as e:
                logger.error(f"Connection Reset Error for search query '{query}': {e}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying search for query '{query}' ({retry+1}/{max_retries})...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return []
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection Error for search query '{query}': {e}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying search for query '{query}' ({retry+1}/{max_retries})...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return []
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout Error for search query '{query}': {e}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying search for query '{query}' ({retry+1}/{max_retries})...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return []
            except Exception as e:
                logger.error(f"Search request failed for query '{query}' (attempt {retry+1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying search for query '{query}' ({retry+1}/{max_retries})...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return []
        
    def get_html_content(self, uuid: str, plain: bool = False) -> str:
        """
        Get HTML content by UUID

        Note: ChatNoir webcontent server has SSL certificate issues,
        so we use HTTP protocol to access chatnoir-webcontent.chatnoir.eu directly

        Args:
            uuid: Document UUID
            plain: Whether to get plain text version (smaller and faster)
        """
        # Access webcontent server directly (using HTTP to avoid SSL issues)
        webcontent_base = "http://chatnoir-webcontent.chatnoir.eu"

        params = {
            "uuid": uuid,
            "index": INDEX_NAME,
        }
        if plain:
            params["plain"] = ""
        
        # Add retry mechanism with longer timeout
        max_retries = 3
        timeout = 180  # Extended timeout to 180 seconds
        retry_delay = 2  # 2 seconds delay between retries

        for retry in range(max_retries):
            try:
                response = requests.get(
                    webcontent_base,
                    params=params,
                    timeout=timeout,
                    verify=False,  # Skip SSL verification
                    stream=True,  # Use streaming to reduce memory usage
                )

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.warning(f"HTML not found for UUID: {uuid}")
                    return ""
                else:
                    logger.warning(f"Cache API Error {response.status_code} for UUID: {uuid}")
                    # Retry for non-404 errors
                    if retry < max_retries - 1:
                        logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        continue
                    return ""

            except ConnectionResetError as e:
                logger.error(f"Connection reset error for {uuid}: {e}")
                # Connection reset error, retry
                if retry < max_retries - 1:
                    logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return ""
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout error for {uuid}: {e}")
                # Timeout error, retry
                if retry < max_retries - 1:
                    logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return ""
            except Exception as e:
                logger.error(f"Failed to fetch HTML for {uuid}: {e}")
                # Other errors, retry
                if retry < max_retries - 1:
                    logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return ""
        
        return ""

    def close(self):
        self.session.close()
