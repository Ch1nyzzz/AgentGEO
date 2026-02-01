import os
import logging
import hashlib
from typing import Optional

class DataSaver:
    """Manages file storage and paths."""
    def __init__(self, base_dir="outputs"):
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def clean_filename(self, url: str) -> str:
        """Converts URL to a valid filename."""
        # Use MD5 hash as filename to avoid length issues or illegal characters
        hash_object = hashlib.md5(url.encode())
        return hash_object.hexdigest()

    def save(self, content: str, filename: str, ext: str = "html", verbose: bool = False):
        """Generic save method."""
        if not content:
            return

        full_path = os.path.join(self.base_dir, f"{filename}.{ext}")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            if verbose:
                logging.info(f"Saved to {full_path} ({len(content)/1024:.2f} KB)")
        except Exception as e:
            logging.error(f"Save failed {full_path}: {e}")

    def load(self, filename: str, ext: str = "html", verbose: bool = False) -> Optional[str]:
        """Read file."""
        full_path = os.path.join(self.base_dir, f"{filename}.{ext}")
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logging.error(f"Read failed {full_path}: {e}")
        else:
            if verbose:
                logging.info(f"File not found: {full_path}")
        return None
