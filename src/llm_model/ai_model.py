import logging
import numpy as np
from typing import Dict
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import content
import os

logger = logging.getLogger("rag_app")


class AIModel:
    def __init__(self, config: Dict):
        """Initialize AI model with config for Azure OpenAI or Vertex AI."""
        self.config = config
        self.provider = config["llm"]["provider"]
        self.embedding_provider = config["embedding"]["provider"]

        if self.provider == "azure":
            self.llm_client = ChatCompletionsClient(
                endpoint=config["llm"]["azure"]["endpoint"],
                credential=AzureKeyCredential(config["llm"]["azure"]["api_key"])
            )
        elif self.provider == "vertex":
            aiplatform.init(
                project=config["llm"]["vertex"]["project_id"],
                location=config["llm"]["vertex"]["location"]
            )
            self.llm_client = aiplatform.Endpoint(
                f"projects/{config['llm']['vertex']['project_id']}/locations/{config['llm']['vertex']['location']}/endpoints/{config['llm']['vertex']['endpoint']}"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        if self.embedding_provider == "azure":
            self.embedding_client = ChatCompletionsClient(
                endpoint=config["embedding"]["azure"]["endpoint"],
                credential=AzureKeyCredential(config["embedding"]["azure"]["api_key"])
            )
        elif self.embedding_provider == "vertex":
            aiplatform.init(
                project=config["embedding"]["vertex"]["project_id"],
                location=config["embedding"]["vertex"]["location"]
            )
            self.embedding_client = aiplatform.Endpoint(
                f"projects/{config['embedding']['vertex']['project_id']}/locations/{config['embedding']['vertex']['location']}/endpoints/{config['embedding']['vertex']['endpoint']}"
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def generate_text(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        """Generate text using the configured LLM."""
        try:
            if self.provider == "azure":
                response = self.llm_client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config["llm"]["azure"]["deployment"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    extra_query_params={"api-version": self.config["llm"]["azure"]["api_version"]}
                )
                return response.choices[0].message.content
            elif self.provider == "vertex":
                response = self.llm_client.predict(
                    instances=[{
                        "content": prompt,
                        "parameters": {
                            "temperature": temperature,
                            "maxOutputTokens": max_tokens,
                            "topP": top_p
                        }
                    }]
                )
                return response.predictions[0]["content"]
        except Exception as e:
            logger.error(f"Failed to generate text with {self.provider}", exc_info=True)
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using the configured embedding provider."""
        try:
            if self.embedding_provider == "azure":
                response = self.embedding_client.embeddings.create(
                    input=[text],
                    model=self.config["embedding"]["azure"]["deployment"],
                    extra_query_params={"api-version": self.config["embedding"]["azure"]["api_version"]}
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            elif self.embedding_provider == "vertex":
                response = self.embedding_client.predict(
                    instances=[{"content": text}]
                )
                return np.array(response.predictions[0]["embeddings"]["values"], dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding with {self.embedding_provider}", exc_info=True)
            raise