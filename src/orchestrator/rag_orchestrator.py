import logging
import yaml
import jsonschema
from src.file_processor.processor_factory import ProcessorFactory
from src.indexing.faiss_context import FaissManager
from src.persistence.oracle_db_manager import OracleDBManager
from src.llm_model.ai_model import AIModel
from src.llm_model.AIRagModel import AIRagModel
import numpy as np

logger = logging.getLogger("rag_app")

class RagOrchestrator:
    def __init__(self, config: Dict):
        """Initialize RAG orchestrator with configuration."""
        self.config = config
        self.ai_model = AIModel(config)
        self.faiss_manager = FaissManager(config, self.ai_model)
        self.db_manager = OracleDBManager(config) if config["persistence"]["persist_metadata"] else None
        self.processor_factory = ProcessorFactory(config, self.ai_model)
        self.rag_model = AIRagModel(config, self.ai_model, self.faiss_manager)

    def load_catalog(self):
        """Load and validate catalog.yaml."""
        with open("config/catalog.yaml", 'r') as f:
            catalog = yaml.safe_load(f)
        self.validate_catalog(catalog)
        return catalog

    def validate_catalog(self, catalog: Dict):
        """Validate catalog.yaml schema."""
        schema = {
            "type": "object",
            "properties": {
                "application_name": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["path", "type", "module"],
                                    "properties": {
                                        "path": {"type": "string"},
                                        "type": {"type": "string"},
                                        "module": {"type": "array"}
                                    }
                                }
                            }
                        },
                        "required": ["name", "files"]
                    }
                }
            },
            "required": ["application_name"]
        }
        try:
            jsonschema.validate(catalog, schema)
        except jsonschema.ValidationError as e:
            logger.error("Invalid catalog.yaml schema", exc_info=True)
            raise

    def process_files(self):
        """Process files from catalog and index them."""
        catalog = self.load_catalog()
        for app in catalog["application_name"]:
            for file_info in app["files"]:
                try:
                    processor = self.processor_factory.get_processor(file_info["type"])
                    metadata = processor.process_file(file_info["path"], app["name"], file_info)
                    embeddings = np.array([self.ai_model.embed_text(m["chunk_text"]) for m in metadata])
                    self.faiss_manager.add_to_index(embeddings, metadata)
                    if self.db_manager:
                        self.db_manager.insert_metadata(metadata)
                except Exception as e:
                    logger.error(f"Failed to process file: {file_info['path']}", exc_info=True)

    def run_queries(self):
        """Run predefined queries."""
        queries = [
            ("Find all employees in HR schema", ["project_alpha"]),
            ("Calculate total sales for Q1 2024", ["project_beta"]),
            ("Summarize the project_beta balance sheet", ["project_beta"]),
            ("Describe the project_alpha logo", ["project_alpha"]),
            ("Give me all hard stop rules", ["project_beta"])
        ]
        for query, app_names in queries:
            try:
                response = self.rag_model.ask(query, app_names)
                logger.info(f"Query: {query}\nResponse: {response}\n")
            except Exception as e:
                logger.error(f"Failed to process query: {query}", exc_info=True)

    def cleanup(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()

    def run(self):
        """Run the RAG pipeline."""
        try:
            self.process_files()
            self.run_queries()
        finally:
            self.cleanup()