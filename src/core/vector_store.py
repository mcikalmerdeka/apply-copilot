import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config.logging_config import setup_logger
from src.config.settings import (
    EMBEDDING_MODEL, 
    TOP_K_RESULTS, 
    PORTFOLIO_TOP_K,
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    PORTFOLIO_CHUNK_SIZE,
    PORTFOLIO_CHUNK_OVERLAP
)

logger = setup_logger(__name__)


class VectorStoreManager:
    """Manages FAISS vector store for resume and portfolio documents with hybrid approach."""
    
    def __init__(self, embeddings_model: str = EMBEDDING_MODEL):
        """
        Initialize the vector store manager.
        
        Args:
            embeddings_model: OpenAI embeddings model name
        """
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.resume_vector_store = None
        self.portfolio_vector_store = None
        self.resume_text_cache = None  # For direct injection when resume is short
        logger.info(f"Initialized VectorStoreManager with embeddings model: {embeddings_model}")
    
    def load_and_index_resume(self, resume_path: str) -> Dict[str, Any]:
        """
        Load a resume PDF and create/update the FAISS vector store.
        Returns resume info including text length for hybrid approach decision.
        
        Args:
            resume_path: Path to the resume PDF file
            
        Returns:
            Dictionary with resume info (use_direct_injection, text_length)
        """
        try:
            logger.info(f"Loading resume from: {resume_path}")
            
            # Load PDF
            loader = PyPDFLoader(resume_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from resume")
            
            # Combine all text for length check
            full_text = "\n\n".join([doc.page_content for doc in documents])
            text_length = len(full_text)
            logger.info(f"Resume total length: {text_length} characters")
            
            # Always cache the full text for potential direct injection
            self.resume_text_cache = full_text
            
            # Also create vector store for potential future use
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"Split resume into {len(splits)} chunks")
            
            # Create or update vector store
            if self.resume_vector_store is None:
                self.resume_vector_store = FAISS.from_documents(splits, self.embeddings)
                logger.info("Created new FAISS vector store for resume")
            else:
                self.resume_vector_store.add_documents(splits)
                logger.info("Added documents to existing resume vector store")
            
            return {
                "use_direct_injection": True,  # Always prefer direct for resume
                "text_length": text_length,
                "chunks_created": len(splits)
            }
                
        except Exception as e:
            logger.error(f"Error loading and indexing resume: {str(e)}")
            raise
    
    def load_and_index_portfolio(self, portfolio_path: str) -> Dict[str, Any]:
        """
        Load a portfolio text file and create/update the FAISS vector store.
        Portfolio uses RAG approach due to longer content.
        
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
            
            # Create or update portfolio vector store
            if self.portfolio_vector_store is None:
                self.portfolio_vector_store = FAISS.from_documents(splits, self.embeddings)
                logger.info("Created new FAISS vector store for portfolio")
            else:
                self.portfolio_vector_store.add_documents(splits)
                logger.info("Added documents to existing portfolio vector store")
            
            return {
                "text_length": text_length,
                "chunks_created": len(splits),
                "use_rag": True
            }
                
        except Exception as e:
            logger.error(f"Error loading and indexing portfolio: {str(e)}")
            raise
    
    def get_resume_context(self, use_rag: bool = False, query: str = None, k: int = TOP_K_RESULTS) -> str:
        """
        Get resume context either through direct injection or RAG.
        
        Args:
            use_rag: If True, use RAG retrieval; if False, return full text
            query: Search query (required if use_rag=True)
            k: Number of chunks to retrieve if using RAG
            
        Returns:
            Resume context as string
        """
        if not use_rag and self.resume_text_cache:
            logger.info("Using direct resume injection (full text)")
            return self.resume_text_cache
        
        if use_rag and query and self.resume_vector_store:
            logger.info(f"Using RAG for resume retrieval with k={k}")
            results = self.resume_vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in results])
            return context
        
        if self.resume_text_cache:
            logger.info("Falling back to direct resume injection")
            return self.resume_text_cache
        
        raise ValueError("No resume data available. Please load a resume first.")
    
    def get_portfolio_context(self, query: str, k: int = PORTFOLIO_TOP_K) -> str:
        """
        Get portfolio context through RAG retrieval.
        Portfolio always uses RAG due to length.
        
        Args:
            query: Search query (job description)
            k: Number of chunks to retrieve
            
        Returns:
            Portfolio context as string
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
        """Check if resume has been indexed."""
        return self.resume_text_cache is not None
    
    def save_vector_store(self, save_path: str, store_type: str = "resume") -> None:
        """
        Save a FAISS vector store to disk.
        
        Args:
            save_path: Directory path to save the vector store
            store_type: Either "resume" or "portfolio"
        """
        vector_store = self.resume_vector_store if store_type == "resume" else self.portfolio_vector_store
        
        if vector_store is None:
            logger.warning(f"No {store_type} vector store to save")
            return
        
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            vector_store.save_local(save_path)
            logger.info(f"Saved {store_type} vector store to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving {store_type} vector store: {str(e)}")
            raise
    
    def load_vector_store(self, load_path: str, store_type: str = "resume") -> None:
        """
        Load a FAISS vector store from disk.
        
        Args:
            load_path: Directory path containing the saved vector store
            store_type: Either "resume" or "portfolio"
        """
        try:
            vector_store = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            if store_type == "resume":
                self.resume_vector_store = vector_store
            else:
                self.portfolio_vector_store = vector_store
                
            logger.info(f"Loaded {store_type} vector store from: {load_path}")
        except Exception as e:
            logger.error(f"Error loading {store_type} vector store: {str(e)}")
            raise
    
    def clear_vector_store(self, store_type: str = "all") -> None:
        """
        Clear vector stores from memory.
        
        Args:
            store_type: "resume", "portfolio", or "all"
        """
        if store_type in ["resume", "all"]:
            self.resume_vector_store = None
            self.resume_text_cache = None
            logger.info("Resume vector store cleared")
        
        if store_type in ["portfolio", "all"]:
            self.portfolio_vector_store = None
            logger.info("Portfolio vector store cleared")
    
    def get_resume_retriever(self, k: int = TOP_K_RESULTS):
        """
        Get a retriever for the resume vector store.
        
        Args:
            k: Number of top results to return
            
        Returns:
            Retriever object
        """
        if self.resume_vector_store is None:
            logger.error("Resume vector store not initialized")
            raise ValueError("Resume vector store not initialized. Please load or create a vector store first.")
        
        return self.resume_vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_portfolio_retriever(self, k: int = PORTFOLIO_TOP_K):
        """
        Get a retriever for the portfolio vector store.
        
        Args:
            k: Number of top results to return
            
        Returns:
            Retriever object
        """
        if self.portfolio_vector_store is None:
            logger.error("Portfolio vector store not initialized")
            raise ValueError("Portfolio vector store not initialized. Please load or create a vector store first.")
        
        return self.portfolio_vector_store.as_retriever(search_kwargs={"k": k})
