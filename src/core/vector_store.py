from pathlib import Path
from typing import Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.config.logging_config import setup_logger
from src.config.settings import (
    EMBEDDING_MODEL,
    PORTFOLIO_TOP_K,
    PORTFOLIO_CHUNK_SIZE,
    PORTFOLIO_CHUNK_OVERLAP
)

logger = setup_logger(__name__)


class VectorStoreManager:
    """Manages context for resume (direct injection) and portfolio (FAISS RAG)."""

    def __init__(self, embeddings_model: str = EMBEDDING_MODEL):
        """
        Initialize the vector store manager.

        Args:
            embeddings_model: OpenAI embeddings model name
        """
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.portfolio_vector_store = None
        self.resume_text_cache = None  # Resume uses direct injection (full text)
        logger.info(f"Initialized VectorStoreManager with embeddings model: {embeddings_model}")

    def load_and_index_resume(self, resume_path: str) -> Dict[str, Any]:
        """
        Load a resume PDF and cache its full text for direct injection.

        The resume is a short 1-2 page document, so it is never indexed/embedded -
        the full text is injected directly into the LLM context window.

        Args:
            resume_path: Path to the resume PDF file

        Returns:
            Dictionary with resume info (text_length)
        """
        try:
            logger.info(f"Loading resume from: {resume_path}")

            loader = PyPDFLoader(resume_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from resume")

            full_text = "\n\n".join([doc.page_content for doc in documents])
            text_length = len(full_text)
            logger.info(f"Resume total length: {text_length} characters")

            self.resume_text_cache = full_text

            return {"text_length": text_length}

        except Exception as e:
            logger.error(f"Error loading resume: {str(e)}")
            raise

    def load_and_index_portfolio(self, portfolio_path: str) -> Dict[str, Any]:
        """
        Load a portfolio text file and create/update the FAISS vector store (RAG).

        Args:
            portfolio_path: Path to the portfolio text file

        Returns:
            Dictionary with portfolio info
        """
        try:
            logger.info(f"Loading portfolio from: {portfolio_path}")

            # Load text file
            loader = TextLoader(portfolio_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Loaded portfolio document")

            # Combine all text for length info
            full_text = "\n\n".join([doc.page_content for doc in documents])
            text_length = len(full_text)
            logger.info(f"Portfolio total length: {text_length} characters")

            # Use larger chunks for portfolio (more content per chunk)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=PORTFOLIO_CHUNK_SIZE,
                chunk_overlap=PORTFOLIO_CHUNK_OVERLAP,
                separators=["\n===", "\n---", "\n\n", "\n", " "]  # Prioritize section breaks
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"Split portfolio into {len(splits)} chunks")

            # Create or update FAISS vector store (RAG)
            if self.portfolio_vector_store is None:
                self.portfolio_vector_store = FAISS.from_documents(splits, self.embeddings)
                logger.info("Created new FAISS vector store for portfolio")
            else:
                self.portfolio_vector_store.add_documents(splits)
                logger.info("Added documents to existing portfolio vector store")

            return {
                "text_length": text_length,
                "chunks_created": len(splits)
            }

        except Exception as e:
            logger.error(f"Error loading and indexing portfolio: {str(e)}")
            raise

    def get_resume_context(self) -> str:
        """
        Get resume context via direct injection (full cached text).

        Returns:
            Resume context as string
        """
        if self.resume_text_cache:
            logger.info("Using direct resume injection (full text)")
            return self.resume_text_cache

        raise ValueError("No resume data available. Please load a resume first.")

    def get_portfolio_context(self, query: str, k: int = PORTFOLIO_TOP_K) -> str:
        """
        Get portfolio context through RAG retrieval.

        Args:
            query: Search query (job description)
            k: Number of chunks to retrieve

        Returns:
            Portfolio context as string, or empty string if no portfolio indexed
        """
        if self.portfolio_vector_store is None:
            logger.info("No portfolio indexed, returning empty context")
            return ""

        logger.info(f"Retrieving portfolio context with k={k}")
        results = self.portfolio_vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        logger.info(f"Retrieved {len(results)} portfolio chunks")
        return context

    def has_portfolio(self) -> bool:
        """Check if portfolio has been indexed."""
        return self.portfolio_vector_store is not None

    def has_resume(self) -> bool:
        """Check if resume has been loaded."""
        return self.resume_text_cache is not None

    def build_context(self, query: str) -> str:
        """
        Build the combined generation context using the hybrid approach:
        - Resume: Direct injection (full text, no RAG)
        - Portfolio: RAG retrieval (if available)

        Args:
            query: Query used for portfolio RAG retrieval (job description or question)

        Returns:
            Combined context string ("" if no resume is loaded)
        """
        context_parts = []

        # 1. Add resume context (always direct injection)
        if self.has_resume():
            resume_context = self.get_resume_context()
            context_parts.append("=== RESUME ===\n" + resume_context)
            logger.info(f"Added resume context ({len(resume_context)} chars)")
        else:
            logger.warning("No resume loaded")

        # 2. Add portfolio context via RAG (if available)
        if self.has_portfolio():
            portfolio_context = self.get_portfolio_context(query)
            if portfolio_context:
                context_parts.append("\n\n=== RELEVANT PROJECTS FROM PORTFOLIO ===\n" + portfolio_context)
                logger.info(f"Added portfolio context via RAG ({len(portfolio_context)} chars)")
        else:
            logger.info("No portfolio loaded (optional)")

        return "\n\n".join(context_parts)

    def save_vector_store(self, save_path: str) -> None:
        """
        Save the portfolio FAISS vector store to disk.

        Args:
            save_path: Directory path to save the vector store
        """
        if self.portfolio_vector_store is None:
            logger.warning("No portfolio vector store to save")
            return

        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            self.portfolio_vector_store.save_local(save_path)
            logger.info(f"Saved portfolio vector store to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving portfolio vector store: {str(e)}")
            raise

    def clear_vector_store(self) -> None:
        """Clear all resume text cache and portfolio vector store from memory."""
        self.resume_text_cache = None
        self.portfolio_vector_store = None
        logger.info("Resume cache and portfolio vector store cleared")
