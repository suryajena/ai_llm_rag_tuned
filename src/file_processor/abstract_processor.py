import logging
from typing import List, Dict
from abc import ABC, abstractmethod
import os

logger = logging.getLogger("rag_app")

class AbstractFileProcessor(ABC):
    def __init__(self, config: Dict=None):
        """Initialize processor with configuration."""
        self.config = config

    @abstractmethod
    def process_file(self, file_path: str, application_name: str, config: Dict) -> List[Dict]:
        """Process a file and return chunked metadata."""
        pass

    def _split_text_into_chunks(self, text: str) -> List[Dict]:
        """Split text into chunks for indexing."""
        chunk_size = self.config.get("chunking", {}).get("size", 1000) if self.config else 1000
        overlap = self.config.get("chunking", {}).get("overlap", 200) if self.config else 200
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append({
                "chunk_text": chunk,
                "start_char": i,
                "end_char": i + len(chunk)
            })
        return chunks

    def _validate_file_path(self, file_path: str) -> None:
        """Validate file path to prevent directory traversal."""
        base_dir = self.config["data_paths"]["input_directory"] if self.config else "data/"
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(os.path.realpath(base_dir)):
            logger.error(f"Invalid file path: {file_path}")
            raise ValueError(f"File path {file_path} is outside input directory")