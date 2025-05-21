import logging
import oracledb
from typing import List, Dict
import json

logger = logging.getLogger("rag_app")

class OracleDBManager:
    def __init__(self, config: Dict):
        """Initialize Oracle DB manager with config."""
        self.config = config
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish Oracle DB connection."""
        try:
            self.conn = oracledb.connect(
                user=self.config["oracle"]["user"],
                password=self.config["oracle"]["password"],
                dsn=self.config["oracle"]["url"]
            )
            logger.info("Connected to Oracle DB")
        except oracledb.Error as e:
            logger.error("Failed to connect to Oracle DB", exc_info=True)
            self.conn = None

    def insert_metadata(self, metadata: List[Dict]):
        """Insert metadata into Oracle DB."""
        if not self.conn:
            logger.warning("No DB connection, skipping metadata insert")
            return
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO pm_rag_metadata (
                        FILE_PATH, APPLICATION_NAME, FILE_TYPE, CHUNK_TEXT, START_CHAR, END_CHAR,
                        PAGE_NUMBER, LAST_UPDATED_DATE, REGION, STATUS, ADDITIONAL_METADATA
                    ) VALUES (:1, :2, :3, :4, :5, :6, :7, TO_DATE(:8, 'YYYY-MM-DD'), :9, :10, :11)
                """, [
                    (
                        meta["file_path"], meta["application_name"], meta["file_type"], meta["chunk_text"],
                        meta["start_char"], meta["end_char"], meta.get("page_number", 0),
                        meta["last_updated_date"], meta["region"], meta["status"],
                        json.dumps(meta.get("additional_metadata", {}))
                    ) for meta in metadata
                ])
                self.conn.commit()
                logger.info(f"Inserted {len(metadata)} metadata records")
        except oracledb.Error as e:
            logger.error("Failed to insert metadata", exc_info=True)
            raise

    def close(self):
        """Close Oracle DB connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed Oracle DB connection")