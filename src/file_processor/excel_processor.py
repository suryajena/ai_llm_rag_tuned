import logging
import pandas as pd
from typing import List, Dict
from src.file_processor.abstract_processor import AbstractFileProcessor
from src.util.helper import FileProcessingError, ExcelCatalogMissingError

logger = logging.getLogger("rag_app")


class ExcelProcessor(AbstractFileProcessor):
    def process_file(self, file_path: str, application_name: str, config: Dict) -> List[Dict]:
        """Process an Excel file and return chunked metadata."""
        try:
            self._validate_file_path(file_path)
            catalog_sheet = config.get("excel_config", {}).get("catalog_sheet_name", "Metadata")
            sheets = pd.ExcelFile(file_path).sheet_names
            if catalog_sheet not in sheets:
                logger.error(f"Catalog sheet {catalog_sheet} not found in {file_path}")
                raise ExcelCatalogMissingError(f"Catalog sheet {catalog_sheet} missing")

            catalog_df = pd.read_excel(file_path, sheet_name=catalog_sheet)
            column_metadata = self._extract_catalog_metadata(catalog_df)

            metadata = []
            for sheet in sheets:
                if sheet == catalog_sheet:
                    continue
                df = pd.read_excel(file_path, sheet_name=sheet, nrows=0)
                chunk_text = f"Sheet: {sheet}, Columns: {', '.join(df.columns)}"
                chunks = self._split_text_into_chunks(chunk_text)
                for chunk in chunks:
                    metadata.append({
                        "file_path": file_path,
                        "application_name": application_name,
                        "file_type": "excel",
                        "chunk_text": chunk["chunk_text"],
                        "start_char": chunk["start_char"],
                        "end_char": chunk["end_char"],
                        "page_number": 0,
                        "region": config["region"],
                        "last_updated_date": config["last_updated_date"],
                        "status": config["status"],
                        "additional_metadata": {
                            "sheet_name": sheet,
                            "column_metadata": column_metadata,
                            "description": config.get("description", ""),
                            "modules": config.get("module", [])
                        }
                    })
            logger.info(f"Processed Excel file: {file_path}, {len(metadata)} chunks")
            return metadata
        except (IOError, ValueError) as e:
            logger.error(f"Failed to process Excel file: {file_path}", exc_info=True)
            raise FileProcessingError(f"Error processing {file_path}: {str(e)}")

    def _extract_catalog_metadata(self, catalog_df: pd.DataFrame) -> Dict:
        """Extract column metadata from catalog sheet."""
        try:
            expected_columns = ["Column Name", "Description"]
            if not all(col in catalog_df.columns for col in expected_columns):
                raise ValueError(f"Catalog sheet must have columns: {expected_columns}")
            metadata = {}
            for _, row in catalog_df.iterrows():
                col_name = str(row["Column Name"])
                col_desc = str(row["Description"])
                metadata[col_name] = {"description": col_desc}
            return metadata
        except ValueError as e:
            logger.error("Failed to extract catalog metadata", exc_info=True)
            raise FileProcessingError(f"Error extracting catalog metadata: {str(e)}")