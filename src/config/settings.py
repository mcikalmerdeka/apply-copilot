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
LLM_MODEL = "gpt-4.1-mini"

# Vector store settings
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# Cover letter settings
MAX_WORDS = 500
CANDIDATE_NAME = "Muhammad Cikal Merdeka"

# Contact Links
RESUME_AI_LINK = "https://drive.google.com/file/d/1ie89a22v7E1iMEglhj9KXBgu72YSq2K2/view?usp=sharing"
RESUME_DATA_LINK = "https://drive.google.com/file/d/14vDbB6alxpNIlo266OStm0lHaqusHsxc/view?usp=sharing"
GITHUB_LINK = "https://github.com/mcikalmerdeka"
WEBSITE_LINK = "https://mcikalmerdeka.vercel.app/"
