import logging
import faiss
import numpy as np
from typing import List, Dict, Optional
import json
import os
import yaml

logger = logging.getLogger("rag_app")

class FaissManager:
    def __init__(self, config: Dict, ai_model):
        """Initialize FAISS manager with config and AI model."""
        self.config = config
        self.ai_model = ai_model
        self.dimension = config["embedding"][config["embedding"]["provider"]]["dimension"]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._load_config()

    def _load_config(self):
        """Load configuration from config.yaml."""
        try:
            with open("config/config.yaml", 'r') as f:
                self.config = yaml.safe_load(f)
            self.index_file = self.config["embedding"]["index_file_path"]
            self.id_map_file = self.config["persistence"]["faiss_id_map_path"]
            if self.config["embedding"]["reload_index"] and os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                with open(self.id_map_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded FAISS index from {self.index_file}")
        except (IOError, KeyError) as e:
            logger.error("Failed to load config", exc_info=True)
            raise

    def add_to_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings and metadata to FAISS index."""
        try:
            self.index.add(embeddings)
            self.metadata.extend(metadata)
            faiss.write_index(self.index, self.index_file)
            with open(self.id_map_file, 'w') as f:
                json.dump(self.metadata, f)
            logger.info(f"Added {len(metadata)} items to FAISS index")
        except Exception as e:
            logger.error("Failed to add to FAISS index", exc_info=True)
            raise

    def search(self, query_embedding: np.ndarray, k: int, application_names: Optional[List[str]] = None,
               file_types: Optional[List[str]] = None, regions: Optional[List[str]] = None,
               statuses: Optional[List[str]] = None, modules: Optional[List[str]] = None) -> List[Dict]:
        """Search FAISS index for relevant metadata."""
        try:
            distances, indices = self.index.search(query_embedding, k)
            results = []
            for idx in indices[0]:
                if idx < len(self.metadata):
                    meta = self.metadata[idx]
                    if self._matches_filters(meta, application_names, file_types, regions, statuses, modules):
                        results.append(meta)
            logger.debug(f"FAISS search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error("FAISS search failed", exc_info=True)
            raise

    def _matches_filters(self, meta: Dict, application_names: Optional[List[str]], file_types: Optional[List[str]],
                         regions: Optional[List[str]], statuses: Optional[List[str]], modules: Optional[List[str]]) -> bool:
        """Check if metadata matches filters."""
        return (
            (not application_names or meta["application_name"] in application_names) and
            (not file_types or meta["file_type"] in file_types) and
            (not regions or meta["region"] in regions) and
            (not statuses or meta["status"] in statuses) and
            (not modules or any(m in meta.get("additional_metadata", {}).get("modules", []) for m in modules))
        )