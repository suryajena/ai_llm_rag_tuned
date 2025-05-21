import logging
import json
from PIL import Image
import pytesseract
from typing import List, Dict
from src.file_processor.abstract_processor import AbstractFileProcessor
from src.util.helper import FileProcessingError
from tenacity import retry, stop_after_attempt, wait_fixed
import os
import yaml

logger = logging.getLogger("rag_app")


class ImageProcessor(AbstractFileProcessor):
    def __init__(self, config: Dict = None, ai_model=None):
        """Initialize ImageProcessor with config and AI model."""
        super().__init__(config)
        self.ai_model = ai_model
        self.description_source = config["image_processing"]["description_source"] if config else "metadata"
        self.ocr_language = config["image_processing"]["ocr_language"] if config else "eng"
        self.cache_path = os.path.join(config["data_paths"]["output_directory"],
                                       "image_description_cache.json") if config else "output/image_description_cache.json"

    def process_file(self, file_path: str, application_name: str, config: Dict) -> List[Dict]:
        """Process an image file and return metadata with content based on description_source."""
        try:
            self._validate_file_path(file_path)
            with Image.open(file_path) as img:
                img.verify()
                img = Image.open(file_path)  # Reopen after verify
                chunk_text = self._get_image_content(img, file_path)

            chunks = self._split_text_into_chunks(chunk_text)
            metadata = [
                {
                    "file_path": file_path,
                    "application_name": application_name,
                    "file_type": "image",
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
            logger.info(f"Processed image file: {file_path}, {len(metadata)} chunks")
            return metadata
        except (IOError, ValueError) as e:
            logger.error(f"Failed to process image file: {file_path}", exc_info=True)
            raise FileProcessingError(f"Error processing {file_path}: {str(e)}")

    def _get_image_content(self, img: Image.Image, file_path: str) -> str:
        """Extract content based on description_source."""
        if self.description_source == "llm" and self.ai_model:
            return self._extract_llm_content(file_path)
        elif self.description_source == "ocr":
            return self._extract_ocr_content(img)
        elif self.description_source == "metadata":
            return self._extract_metadata_content(img)
        logger.warning(f"Invalid description_source: {self.description_source}, using metadata")
        return self._extract_metadata_content(img)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _extract_llm_content(self, file_path: str) -> str:
        """Generate image description using LLM with caching."""
        try:
            cache = json.load(open(self.cache_path, 'r')) if os.path.exists(self.cache_path) else {}
            if file_path in cache:
                logger.debug(f"Using cached LLM description for {file_path}")
                return cache[file_path]

            with open(self.config["prompts_path"], 'r') as f:
                prompts = yaml.safe_load(f)
            prompt = prompts[self.config["llm"]["provider"]]["image_description_prompt"]
            description = self.ai_model.generate_text(
                prompt,
                temperature=self.config["llm"]["temperature"],
                max_tokens=self.config["llm"]["max_tokens"],
                top_p=self.config["llm"]["top_p"]
            )
            cache[file_path] = description
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            json.dump(cache, open(self.cache_path, 'w'))
            return description
        except (IOError, KeyError, Exception) as e:
            logger.error(f"Failed to generate LLM description for {file_path}", exc_info=True)
            return "Image content not described (LLM error)"

    def _extract_ocr_content(self, img: Image.Image) -> str:
        """Extract text from image using pytesseract."""
        try:
            text = pytesseract.image_to_string(img, lang=self.ocr_language)
            return text.strip() or "No text detected in image"
        except pytesseract.TesseractError as e:
            logger.error("Failed to extract OCR content", exc_info=True)
            return f"No text detected in image (OCR error: {str(e)})"

    def _extract_metadata_content(self, img: Image.Image) -> str:
        """Extract image metadata using Pillow."""
        try:
            metadata = [
                f"Format: {img.format}",
                f"Size: {img.size[0]}x{img.size[1]}",
                f"Mode: {img.mode}"
            ]
            exif = img.getexif()
            if exif:
                metadata.append(f"EXIF: {dict(exif)}")
            return "; ".join(metadata)
        except Exception as e:
            logger.error("Failed to extract metadata", exc_info=True)
            return f"No metadata extracted: {str(e)}"