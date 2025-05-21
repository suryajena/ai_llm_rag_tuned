import logging
import yaml
from src.orchestrator.rag_orchestrator import RagOrchestrator

logger = logging.getLogger("rag_app")

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open("config/config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except IOError as e:
        logger.error("Failed to load config.yaml", exc_info=True)
        raise

def main():
    """Main entry point for RAG system."""
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    orchestrator = RagOrchestrator(config)
    orchestrator.run()

if __name__ == "__main__":
    main()