import os
from pathlib import Path

# Base directory (project root: src/config/settings.py -> go up to project)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
COVER_LETTER_EXAMPLES_DIR = DATA_DIR / "cover_letter_examples"
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure directories exist
for directory in [DATA_DIR, RESUMES_DIR, COVER_LETTER_EXAMPLES_DIR, VECTOR_STORES_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "glm-5.3-flash"  # OpenCode OpenAI-compatible endpoint

# Vector store settings - Optimized for Portfolio RAG
CHUNK_SIZE = 800  # Increased for larger semantic units in portfolio
CHUNK_OVERLAP = 100  # Increased overlap for better context continuity
TOP_K_RESULTS = 5  # Increased to get more relevant project details

# Portfolio settings
PORTFOLIO_CHUNK_SIZE = 1000
PORTFOLIO_CHUNK_OVERLAP = 150
PORTFOLIO_TOP_K = 8  # Get more results for comprehensive portfolio coverage

# Cover letter settings
MAX_WORDS = 500
CANDIDATE_NAME = "Muhammad Cikal Merdeka"

# Contact Links
GITHUB_LINK = "https://github.com/mcikalmerdeka"
WEBSITE_LINK = "https://mcikalmerdeka.vercel.app/"
