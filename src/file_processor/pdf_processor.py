import logging
import PyPDF2
from typing import List, Dict
from src.file_processor.abstract_processor import AbstractFileProcessor
from src.util.helper import FileProcessingError

logger = logging.getLogger("rag_app")

class PdfProcessor(AbstractFileProcessor):
    def process_file(self, file_path: str, application_name: str, config: Dict) -> List[Dict]:
        """Process a PDF file and return chunked metadata."""
        try:
            self._validate_file_path(file_path)
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                metadata = []
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text() or ""
                    chunks = self._split_text_into_chunks(text)
                    for chunk in chunks:
                        metadata.append({
                            "file_path": file_path,
                            "application_name": application_name,
                            "file_type": "pdf",
                            "chunk_text": chunk["chunk_text"],
                            "start_char": chunk["start_char"],
                            "end_char": chunk["end_char"],
                            "page_number": page_num + 1,
                            "region": config["region"],
                            "last_updated_date": config["last_updated_date"],
                            "status": config["status"],
                            "additional_metadata": {
                                "description": config.get("description", ""),
                                "modules": config.get("module", [])
                            }
                        })
            logger.info(f"Processed PDF file: {file_path}, {len(metadata)} chunks")
            return metadata
        except (IOError, ValueError) as e:
            logger.error(f"Failed to process PDF file: {file_path}", exc_info=True)
            raise FileProcessingError(f"Error processing {file_path}: {str(e)}")