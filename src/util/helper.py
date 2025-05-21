class FileProcessingError(Exception):
    """Exception raised for file processing errors."""
    pass

class ExcelCatalogMissingError(Exception):
    """Exception raised when Excel catalog sheet is missing."""
    pass

class ContextMismatchError(Exception):
    """Exception raised when no relevant context is found."""
    pass