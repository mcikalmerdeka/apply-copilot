import os
from typing import Optional
from dotenv import load_dotenv
from tavily import TavilyClient

from src.config.logging_config import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class WebSearchTool:
    """Tavily-based web search tool for retrieving real-time market information."""

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.warning("TAVILY_API_KEY not found in environment. Web search will be disabled.")
            self.client = None
        else:
            self.client = TavilyClient(api_key=api_key)
            logger.info("Initialized Tavily web search client")

    def is_salary_question(self, question: str) -> bool:
        """Detect if a question is asking about salary or compensation."""
        keywords = [
            "salary", "expected salary", "compensation", "pay", "remuneration",
            "package", "how much do you expect", "expected pay", "salary expectation",
            "monthly salary", "yearly salary", "annual salary", "base salary",
            "take home", "gross salary", "net salary"
        ]
        q = question.lower()
        return any(k in q for k in keywords)

    def search(self, query: str, max_results: int = 3) -> str:
        """Perform a web search and return formatted results."""
        if not self.client:
            logger.warning("Tavily client not initialized. Skipping web search.")
            return ""

        try:
            logger.info(f"Performing web search: {query}")
            response = self.client.search(query, max_results=max_results)
            results = response.get("results", [])
            if not results:
                logger.info("No search results found")
                return ""

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                content = r.get("content", "No content")
                url = r.get("url", "")
                formatted.append(f"[{i}] {title}\n{content}\nSource: {url}")

            return "\n\n".join(formatted)

        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return ""

    def search_salary(self, job_title: str, company_name: Optional[str] = None) -> str:
        """Search for current salary range for a specific job."""
        query_parts = [job_title, "salary range", "2026"]
        if company_name:
            query_parts.insert(1, f"at {company_name}")
        query = " ".join(query_parts)
        return self.search(query, max_results=3)

    def search_general(self, question: str, max_results: int = 3) -> str:
        """Perform a general web search for any question."""
        return self.search(question, max_results=max_results)
