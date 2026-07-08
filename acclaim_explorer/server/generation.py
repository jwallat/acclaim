"""RAG answer generation for the acclaim explorer server."""

from __future__ import annotations

from acclaim.data_models import Document
from acclaim.llm_client import LiteLLMClient

from .config import GeneratorConfig

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided documents.\n"
    "Always cite your sources inline using [N] notation immediately after the relevant statement "
    '(e.g. "Paris is the capital of France [1].").\n'
    "Cite at least one document per factual claim. Only cite documents from the list provided.\n"
    "If the documents don't contain enough information to fully answer the question, say so clearly."
)

_USER_TEMPLATE = (
    "Documents:\n{docs}\n\nQuestion: {question}\n\nAnswer (with inline [N] citations):"
)


class RagGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        self._client = LiteLLMClient(
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self._system = config.system_prompt or _DEFAULT_SYSTEM_PROMPT

    def generate(self, question: str, documents: list[Document]) -> str:
        numbered_docs = "\n\n".join(
            f"[{i}] {doc.text}" for i, doc in enumerate(documents, 1)
        )
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": _USER_TEMPLATE.format(docs=numbered_docs, question=question)},
        ]
        return self._client.call(messages)
