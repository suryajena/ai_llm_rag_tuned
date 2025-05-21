import logging
from typing import List, Dict
from src.file_processor.abstract_processor import AbstractFileProcessor
from src.util.helper import FileProcessingError

logger = logging.getLogger("rag_app")

class TxtProcessor(AbstractFileProcessor):
    def process_file(self, file_path: str, application_name: str, config: Dict) -> List[Dict]:
        """Process a text file and return chunked metadata."""
        try:
            self._validate_file_path(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = self._split_text_into_chunks(text)
            metadata = [
                {
                    "file_path": file_path,
                    "application_name": application_name,
                    "file_type": "txt",
                    "chunk_text": chunk["chunk_text"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "page_number": 0,
                    "region": config["region"],
                    "last_updated_date": config["last_updated_date"],
                    "status": config["status"],
                    "additional_metadata": {
                        "description": config.get("description", ""),
                        "modules": config.get("module", [])
                    }
                } for chunk in chunks
            ]
            logger.info(f"Processed text file: {file_path}, {len(metadata)} chunks")
            return metadata
        except IOError as e:
            logger.error(f"Failed to process text file: {file_path}", exc_info=True)
            raise FileProcessingError(f"Error processing {file_path}: {str(e)}")