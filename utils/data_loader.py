"""
Data Loader

Supports loading documents from multiple data sources:
- Parquet files
- JSON files
- CSV files
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified Data Loader

    Supports loading data with the following fields:
    - raw_html: Original HTML
    - train_queries: List of training queries
    - test_queries: List of test queries (optional)
    - doc_id: Document ID
    - url: Document URL (optional)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Data configuration
                - input_type: Input type (parquet, json, csv)
                - input_path: Input file path
                - required_fields: List of required fields
        """
        self.config = config
        self.input_path = Path(config["input_path"])
        self.input_type = config.get("input_type", "parquet")
        self.required_fields = config.get("required_fields", ["raw_html", "train_queries"])
        self.doc_limit = config.get("doc_limit", None)
        self.doc_offset = config.get("doc_offset", 0)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def load(self) -> List[Dict[str, Any]]:
        """Load data"""
        if self.input_type == "parquet":
            documents = self._load_parquet()
        elif self.input_type == "json":
            documents = self._load_json()
        elif self.input_type == "csv":
            documents = self._load_csv()
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")

        if self.doc_offset:
            documents = documents[self.doc_offset:]
        if self.doc_limit and self.doc_limit > 0:
            documents = documents[:self.doc_limit]
        logger.info(f"After slicing: {len(documents)} documents (offset={self.doc_offset}, limit={self.doc_limit})")
        return documents

    def _load_parquet(self) -> List[Dict[str, Any]]:
        """Load Parquet file"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load parquet files")

        df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(df)} rows from {self.input_path}")

        # Convert to list of dictionaries
        documents = df.to_dict(orient="records")

        # Validate required fields
        self._validate_documents(documents)

        return documents

    def _load_json(self) -> List[Dict[str, Any]]:
        """Load JSON file"""
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support two formats: single document or list of documents
        if isinstance(data, dict):
            documents = [data]
        else:
            documents = data

        logger.info(f"Loaded {len(documents)} documents from {self.input_path}")

        # Validate required fields
        self._validate_documents(documents)

        return documents

    def _load_csv(self) -> List[Dict[str, Any]]:
        """Load CSV file"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load CSV files")

        df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(df)} rows from {self.input_path}")

        # Convert to list of dictionaries
        documents = df.to_dict(orient="records")

        # Validate required fields
        self._validate_documents(documents)

        return documents

    def _validate_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Validate that documents contain required fields"""
        for i, doc in enumerate(documents):
            missing_fields = [field for field in self.required_fields if field not in doc]
            if missing_fields:
                raise ValueError(
                    f"Document {i} is missing required fields: {missing_fields}"
                )

            # Convert numpy arrays to lists
            for field_name in ("train_queries", "test_queries"):
                if field_name in doc and hasattr(doc[field_name], 'tolist'):
                    doc[field_name] = doc[field_name].tolist()

            # Ensure train_queries is a list
            if "train_queries" in doc and isinstance(doc["train_queries"], str):
                try:
                    doc["train_queries"] = json.loads(doc["train_queries"])
                except json.JSONDecodeError:
                    doc["train_queries"] = [doc["train_queries"]]

            # Ensure test_queries is a list (if present)
            if "test_queries" in doc and isinstance(doc["test_queries"], str):
                try:
                    doc["test_queries"] = json.loads(doc["test_queries"])
                except json.JSONDecodeError:
                    doc["test_queries"] = [doc["test_queries"]]

        logger.info(f"Validated {len(documents)} documents")
