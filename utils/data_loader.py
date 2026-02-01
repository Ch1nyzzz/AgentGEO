"""
数据加载器

支持从多种数据源加载文档：
- Parquet 文件
- JSON 文件
- CSV 文件
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DataLoader:
    """
    统一数据加载器

    支持加载包含以下字段的数据：
    - raw_html: 原始 HTML
    - train_queries: 训练查询列表
    - test_queries: 测试查询列表（可选）
    - doc_id: 文档 ID
    - url: 文档 URL（可选）
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 数据配置
                - input_type: 输入类型 (parquet, json, csv)
                - input_path: 输入文件路径
                - required_fields: 必需字段列表
        """
        self.config = config
        self.input_path = Path(config["input_path"])
        self.input_type = config.get("input_type", "parquet")
        self.required_fields = config.get("required_fields", ["raw_html", "train_queries"])

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def load(self) -> List[Dict[str, Any]]:
        """加载数据"""
        if self.input_type == "parquet":
            return self._load_parquet()
        elif self.input_type == "json":
            return self._load_json()
        elif self.input_type == "csv":
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")

    def _load_parquet(self) -> List[Dict[str, Any]]:
        """加载 Parquet 文件"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load parquet files")

        df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(df)} rows from {self.input_path}")

        # 转换为字典列表
        documents = df.to_dict(orient="records")

        # 验证必需字段
        self._validate_documents(documents)

        return documents

    def _load_json(self) -> List[Dict[str, Any]]:
        """加载 JSON 文件"""
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 支持两种格式：单个文档或文档列表
        if isinstance(data, dict):
            documents = [data]
        else:
            documents = data

        logger.info(f"Loaded {len(documents)} documents from {self.input_path}")

        # 验证必需字段
        self._validate_documents(documents)

        return documents

    def _load_csv(self) -> List[Dict[str, Any]]:
        """加载 CSV 文件"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load CSV files")

        df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(df)} rows from {self.input_path}")

        # 转换为字典列表
        documents = df.to_dict(orient="records")

        # 验证必需字段
        self._validate_documents(documents)

        return documents

    def _validate_documents(self, documents: List[Dict[str, Any]]) -> None:
        """验证文档是否包含必需字段"""
        for i, doc in enumerate(documents):
            missing_fields = [field for field in self.required_fields if field not in doc]
            if missing_fields:
                raise ValueError(
                    f"Document {i} is missing required fields: {missing_fields}"
                )

            # 确保 train_queries 是列表
            if "train_queries" in doc and isinstance(doc["train_queries"], str):
                try:
                    doc["train_queries"] = json.loads(doc["train_queries"])
                except json.JSONDecodeError:
                    doc["train_queries"] = [doc["train_queries"]]

            # 确保 test_queries 是列表（如果存在）
            if "test_queries" in doc and isinstance(doc["test_queries"], str):
                try:
                    doc["test_queries"] = json.loads(doc["test_queries"])
                except json.JSONDecodeError:
                    doc["test_queries"] = [doc["test_queries"]]

        logger.info(f"Validated {len(documents)} documents")
