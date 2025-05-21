import logging
import yaml
import json
import re
import pandas as pd
import ast
from typing import List, Dict, Optional
from src.util.helper import ContextMismatchError

logger = logging.getLogger("rag_app")


class AIRagModel:
    def __init__(self, config: Dict, ai_model, faiss_manager):
        """Initialize AIRagModel with config, AI model, and FAISS manager."""
        self.config = config
        self.ai_model = ai_model
        self.faiss_manager = faiss_manager
        self.provider = config["llm"]["provider"]
        prompt_path = config.get("prompts_path", "config/prompts.yaml")
        with open(prompt_path, 'r') as f:
            self.prompts = yaml.safe_load(f)[self.provider]
        logger.info(f"Loaded {self.provider} prompts from {prompt_path}")

    def ask(self, query: str, application_names: Optional[List[str]] = None) -> str:
        """Process a user query and return a response."""
        try:
            intent = self._identify_query_intent(query, application_names)
            query_embedding = self.ai_model.embed_text(query)
            context_chunks = self.faiss_manager.search(
                query_embedding,
                self.config["llm"]["retrieval_k"],
                application_names=intent.get("application_names", application_names),
                file_types=intent.get("file_types"),
                regions=intent.get("regions"),
                statuses=intent.get("statuses"),
                modules=intent.get("modules")
            )
            context_chunks = self._apply_heuristic_filter(query, context_chunks)
            if not context_chunks:
                logger.warning("No relevant context found after filtering")
                raise ContextMismatchError("No relevant context found for query")

            if "sql" in intent.get("file_types", []):
                response = self._handle_sql_query(query, context_chunks)
            elif "excel" in intent.get("file_types", []):
                response = self._handle_excel_query(query, context_chunks, intent.get("analysis_type"))
            else:
                response = self._handle_general_query(query, context_chunks)
            logger.info(f"Generated response for query: {query}")
            return response
        except Exception as e:
            logger.error(f"Failed to process query: {query}", exc_info=True)
            raise

    def _identify_query_intent(self, query: str, application_names: Optional[List[str]]) -> Dict:
        """Classify query intent and return relevant metadata filters."""
        prompt = self.prompts["query_intent_classification_prompt"].format(user_query=query)
        response = self.ai_model.generate_text(
            prompt,
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
            top_p=self.config["llm"]["top_p"]
        )
        try:
            intent = json.loads(response)
            intent["application_names"] = application_names if application_names else intent.get("application_names",
                                                                                                 [])
            logger.debug(f"Query intent: {intent}")
            return intent
        except json.JSONDecodeError as e:
            logger.error("Failed to parse intent JSON", exc_info=True)
            raise ValueError(f"Invalid intent classification response: {str(e)}")

    def _apply_heuristic_filter(self, query: str, context_chunks: List[Dict]) -> List[Dict]:
        """Filter context chunks based on query keywords."""
        keywords = re.findall(r'\b\w+\b', query.lower())
        filtered = []
        for chunk in context_chunks:
            chunk_text = chunk["chunk_text"].lower()
            description = chunk.get("additional_metadata", {}).get("description", "").lower()
            if (chunk["file_type"] not in ["sql", "excel"] or
                    any(keyword in chunk_text for keyword in keywords) or
                    any(keyword in description for keyword in keywords)):
                filtered.append(chunk)
        logger.debug(f"Heuristic filter reduced {len(context_chunks)} to {len(filtered)} chunks")
        return filtered

    def _handle_sql_query(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate SQL query using DDL context."""
        context = "\n".join([chunk["chunk_text"] for chunk in context_chunks])
        prompt = self.prompts["sql_query_generation_prompt"].format(user_query=query, context=context)
        return self.ai_model.generate_text(
            prompt,
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
            top_p=self.config["llm"]["top_p"]
        )

    def _validate_pandas_code(self, code: str) -> bool:
        """Validate Pandas code to ensure safe execution."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ["system", "exec", "eval"]:
                        logger.error(f"Unsafe function {node.func.id} detected in Pandas code")
                        return False
            return True
        except SyntaxError as e:
            logger.error(f"Invalid Pandas code syntax: {str(e)}")
            return False

    def _handle_excel_query(self, query: str, context_chunks: List[Dict], analysis_type: str) -> str:
        """Process Excel query by generating and executing Pandas code."""
        try:
            file_path = context_chunks[0]["file_path"]
            sheet_name = context_chunks[0]["additional_metadata"].get("sheet_name", "Sheet1")
            column_metadata = context_chunks[0]["additional_metadata"].get("column_metadata", {})
            prompt = self.prompts["excel_pandas_code_prompt"].format(
                user_query=query, sheet_name=sheet_name, column_metadata=json.dumps(column_metadata)
            )
            pandas_code = self.ai_model.generate_text(
                prompt,
                temperature=self.config["llm"]["temperature"],
                max_tokens=self.config["llm"]["max_tokens"],
                top_p=self.config["llm"]["top_p"]
            )
            logger.debug(f"Generated Pandas code: {pandas_code}")

            if not self._validate_pandas_code(pandas_code):
                raise ValueError("Unsafe Pandas code detected")

            local_vars = {}
            exec(pandas_code, {"pd": pd, "file_path": file_path, "sheet_name": sheet_name}, local_vars)
            result = local_vars.get("result", "No result returned")

            explain_prompt = self.prompts["excel_human_readable_prompt"].format(
                user_query=query, pandas_result=str(result)
            )
            explanation = self.ai_model.generate_text(
                explain_prompt,
                temperature=self.config["llm"]["temperature"],
                max_tokens=self.config["llm"]["max_tokens"],
                top_p=self.config["llm"]["top_p"]
            )
            return explanation
        except Exception as e:
            logger.error(f"Failed to process Excel query: {query}", exc_info=True)
            raise

    def _handle_general_query(self, query: str, context_chunks: List[Dict]) -> str:
        """Handle general queries using context from text, PDF, or image."""
        context = "\n".join([chunk["chunk_text"] for chunk in context_chunks])
        prompt = self.prompts["general_rag_prompt"].format(user_query=query, context=context)
        return self.ai_model.generate_text(
            prompt,
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
            top_p=self.config["llm"]["top_p"]
        )