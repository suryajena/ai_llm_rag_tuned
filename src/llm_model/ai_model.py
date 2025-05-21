import logging
import numpy as np
from typing import Dict
from openai import AzureOpenAI
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from src.util.helper import get_api_key

logger = logging.getLogger("rag_app")

class AIModel:
    def __init__(self, config: Dict):
        """Initialize AI model with config for Azure OpenAI or Vertex AI."""
        self.config = config
        self.provider = config["llm"]["provider"]
        self.embedding_provider = config["embedding"]["provider"]

        if self.provider == "azure":
            api_key = get_api_key("azure", "llm")
            self.llm_client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=config["llm"]["azure"]["endpoint"],
                api_version=config["llm"]["azure"]["api_version"]
            )
        elif self.provider == "vertex":
            credentials = get_api_key("vertex", "llm")
            vertexai.init(
                project=config["llm"]["vertex"]["project_id"],
                location=config["llm"]["vertex"]["location"],
                api_transport="rest",
                api_endpoint=config["llm"]["vertex"]["api_endpoint"],
                credentials=credentials
            )
            self.llm_client = GenerativeModel(config["llm"]["vertex"]["model"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        if self.embedding_provider == "azure":
            api_key = get_api_key("azure", "embedding")
            self.embedding_client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=config["embedding"]["azure"]["endpoint"],
                api_version=config["embedding"]["azure"]["api_version"]
            )
        elif self.embedding_provider == "vertex":
            credentials = get_api_key("vertex", "embedding")
            vertexai.init(
                project=config["embedding"]["vertex"]["project_id"],
                location=config["embedding"]["vertex"]["location"],
                api_transport="rest",
                api_endpoint=config["embedding"]["vertex"]["api_endpoint"],
                credentials=credentials
            )
            self.embedding_client = TextEmbeddingModel.from_pretrained(config["embedding"]["vertex"]["model"])
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def generate_text(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        """Generate text using the configured LLM."""
        try:
            if self.provider == "azure":
                response = self.llm_client.chat.completions.create(
                    model=self.config["llm"]["azure"]["deployment"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                return response.choices[0].message.content
            elif self.provider == "vertex":
                response = self.llm_client.generate_content(
                    contents=prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": top_p
                    }
                )
                return response.text
        except Exception as e:
            logger.error(f"Failed to generate text with {self.provider}", exc_info=True)
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using the configured embedding provider."""
        try:
            if self.embedding_provider == "azure":
                response = self.embedding_client.embeddings.create(
                    input=text,
                    model=self.config["embedding"]["azure"]["deployment"]
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            elif self.embedding_provider == "vertex":
                response = self.embedding_client.get_embeddings([text])
                return np.array(response[0].values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding with {self.embedding_provider}", exc_info=True)
            raise