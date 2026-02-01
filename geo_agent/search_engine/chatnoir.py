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
        # 增加连接池大小，避免 "Connection pool is full" 警告
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

        # 增加重试机制
        max_retries = 3
        retry_delay = 2
        
        for retry in range(max_retries):
            try:
                # 设置会话的SSL验证为False，避免SSL错误
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
            except requests.exceptions.ConnectionResetError as e:
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
        通过 UUID 获取 HTML 内容

        注意：ChatNoir webcontent 服务器的 HTTPS 有 SSL 证书问题，
        因此直接使用 HTTP 协议访问 chatnoir-webcontent.chatnoir.eu

        Args:
            uuid: 文档 UUID
            plain: 是否获取纯文本版本（更小更快）
        """
        # 直接访问 webcontent 服务器（使用 HTTP 避免 SSL 问题）
        webcontent_base = "http://chatnoir-webcontent.chatnoir.eu"

        params = {
            "uuid": uuid,
            "index": INDEX_NAME,
        }
        if plain:
            params["plain"] = ""
        
        # 增加重试机制和更长的超时时间
        max_retries = 3
        timeout = 180  # 增加超时时间到60秒
        retry_delay = 2  # 重试间隔2秒

        for retry in range(max_retries):
            try:
                response = requests.get(
                    webcontent_base,
                    params=params,
                    timeout=timeout,
                    verify=False,  # 忽略 SSL 验证
                    stream=True,  # 使用流传输，减少内存占用
                )

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.warning(f"HTML not found for UUID: {uuid}")
                    return ""
                else:
                    logger.warning(f"Cache API Error {response.status_code} for UUID: {uuid}")
                    # 非404错误，重试
                    if retry < max_retries - 1:
                        logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        continue
                    return ""

            except ConnectionResetError as e:
                logger.error(f"Connection reset error for {uuid}: {e}")
                # 连接重置错误，重试
                if retry < max_retries - 1:
                    logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return ""
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout error for {uuid}: {e}")
                # 超时错误，重试
                if retry < max_retries - 1:
                    logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return ""
            except Exception as e:
                logger.error(f"Failed to fetch HTML for {uuid}: {e}")
                # 其他错误，重试
                if retry < max_retries - 1:
                    logger.info(f"Retrying {retry+1}/{max_retries} in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    continue
                return ""
        
        return ""

    def close(self):
        self.session.close()
