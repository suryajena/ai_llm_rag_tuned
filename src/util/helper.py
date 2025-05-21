import os
import yaml
import logging

logger = logging.getLogger("rag_app")


class FileProcessingError(Exception):
    """Exception raised for file processing errors."""
    pass


class ExcelCatalogMissingError(Exception):
    """Exception raised when Excel catalog sheet is missing."""
    pass


class ContextMismatchError(Exception):
    """Exception raised when no relevant context is found."""
    pass


def get_api_key(provider: str, key_type: str, config_path: str = "config/config.yaml") -> str:
    return "key here"
