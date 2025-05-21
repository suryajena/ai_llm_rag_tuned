import logging
from typing import Dict
from src.file_processor.abstract_processor import AbstractFileProcessor
from src.file_processor.txt_processor import TxtProcessor
from src.file_processor.sql_processor import SqlProcessor
from src.file_processor.excel_processor import ExcelProcessor
from src.file_processor.pdf_processor import PdfProcessor
from src.file_processor.image_processor import ImageProcessor

logger = logging.getLogger("rag_app")

class ProcessorFactory:
    def __init__(self, config: Dict=None, ai_model=None):
        """Initialize processor factory with config and AI model."""
        self.config = config
        self.ai_model = ai_model
        self.processors: Dict[str, AbstractFileProcessor] = {}
        self._register_processors()

    def _register_processors(self):
        """Register file processors."""
        self.processors["txt"] = TxtProcessor(self.config)
        self.processors["sql"] = SqlProcessor(self.config)
        self.processors["excel"] = ExcelProcessor(self.config)
        self.processors["pdf"] = PdfProcessor(self.config)
        self.processors["image"] = ImageProcessor(self.config, self.ai_model)
        logger.info(f"Registered processors: {list(self.processors.keys())}")

    def get_processor(self, file_type: str) -> AbstractFileProcessor:
        """Get processor for a file type."""
        processor = self.processors.get(file_type)
        if not processor:
            logger.error(f"No processor found for file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
        return processor