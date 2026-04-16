import os
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from src.config.logging_config import setup_logger
from src.config.settings import (LLM_MODEL, MAX_WORDS, COVER_LETTER_EXAMPLES_DIR, 
                                 OUTPUT_DIR, CANDIDATE_NAME, GITHUB_LINK, WEBSITE_LINK)
from src.config.prompts import get_cover_letter_prompt, get_cold_message_prompt
from src.core.vector_store import VectorStoreManager

logger = setup_logger(__name__)


class CoverLetterGenerator:
    """Main class for generating personalized cover letters using hybrid approach."""
    
    def __init__(self, llm_model: str = LLM_MODEL):
        """
        Initialize the cover letter generator.
        
        Args:
            llm_model: LLM model name to use
        """
        self.llm = ChatAnthropic(model=llm_model, temperature=0.7)
        self.vector_store_manager = VectorStoreManager()
        self.cover_letter_examples = []
        logger.info(f"Initialized CoverLetterGenerator with LLM model: {llm_model}")
    
    def load_cover_letter_examples(self) -> None:
        """Load all cover letter examples from the examples directory."""
        try:
            examples_dir = Path(COVER_LETTER_EXAMPLES_DIR)
            pdf_files = list(examples_dir.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDF examples found in {examples_dir}")
                return
            
            logger.info(f"Found {len(pdf_files)} cover letter examples")
            
            for pdf_file in pdf_files:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                self.cover_letter_examples.append({
                    'filename': pdf_file.name,
                    'content': documents[0].page_content
                })
                logger.info(f"Loaded example: {pdf_file.name}")
            
            logger.info(f"Successfully loaded {len(self.cover_letter_examples)} cover letter examples")
            
        except Exception as e:
            logger.error(f"Error loading cover letter examples: {str(e)}")
            raise
    
    def load_portfolio(self, portfolio_path: str) -> dict:
        """
        Load portfolio text file for RAG-based context retrieval.
        
        Args:
            portfolio_path: Path to the portfolio text file
            
        Returns:
            Dictionary with portfolio info
        """
        try:
            result = self.vector_store_manager.load_and_index_portfolio(portfolio_path)
            logger.info("Portfolio loaded and indexed successfully")
            return result
        except Exception as e:
            logger.error(f"Error loading portfolio: {str(e)}")
            raise
    
    def _get_combined_examples(self) -> str:
        """Combine all cover letter examples into a single reference text."""
        if not self.cover_letter_examples:
            logger.warning("No cover letter examples loaded")
            return "No examples available."
        
        combined = "\n\n=== EXAMPLE SEPARATOR ===\n\n".join(
            [f"Example from {ex['filename']}:\n{ex['content']}" 
             for ex in self.cover_letter_examples]
        )
        return combined
    
    def _build_context(self, job_description: str) -> str:
        """
        Build context using hybrid approach:
        - Resume: Direct injection (full text, no RAG)
        - Portfolio: RAG retrieval (if available)
        
        Args:
            job_description: Job description for portfolio retrieval
            
        Returns:
            Combined context string
        """
        context_parts = []
        
        # 1. Add resume context (always direct injection)
        if self.vector_store_manager.has_resume():
            resume_context = self.vector_store_manager.get_resume_context(use_rag=False)
            context_parts.append("=== RESUME ===\n" + resume_context)
            logger.info(f"Added resume context ({len(resume_context)} chars)")
        else:
            logger.warning("No resume loaded")
        
        # 2. Add portfolio context via RAG (if available)
        if self.vector_store_manager.has_portfolio():
            portfolio_context = self.vector_store_manager.get_portfolio_context(job_description)
            if portfolio_context:
                context_parts.append("\n\n=== RELEVANT PROJECTS FROM PORTFOLIO ===\n" + portfolio_context)
                logger.info(f"Added portfolio context via RAG ({len(portfolio_context)} chars)")
        else:
            logger.info("No portfolio loaded (optional)")
        
        return "\n\n".join(context_parts)
    
    def generate_cover_letter(self, job_description: str, company_name: str, 
                            job_title: str) -> str:
        """
        Generate a personalized cover letter using hybrid context approach.
        
        Args:
            job_description: The job description text
            company_name: Name of the company
            job_title: Title of the position
        
        Returns:
            Generated cover letter text
        """
        try:
            logger.info(f"Generating cover letter for {job_title} at {company_name}")
            
            # Get the prompt template
            template = get_cover_letter_prompt(MAX_WORDS)
            prompt = ChatPromptTemplate.from_template(template)
            
            # Get combined examples
            examples_text = self._get_combined_examples()
            
            # Build hybrid context (resume direct + portfolio RAG)
            context = self._build_context(job_description)
            
            # Generate the cover letter
            messages = prompt.format_messages(
                context=context,
                job_description=job_description,
                example_style=examples_text,
                candidate_name=CANDIDATE_NAME,
                max_words=MAX_WORDS
            )
            
            result = self.llm.invoke(messages)
            logger.info("Cover letter generated successfully")
            
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating cover letter: {str(e)}")
            raise
    
    def save_cover_letter(self, cover_letter: str, company_name: str, 
                          job_title: str, format: str = "txt") -> str:
        """
        Save the generated cover letter to a file.
        
        Args:
            cover_letter: The generated cover letter text
            company_name: Name of the company
            job_title: Title of the position
            format: Output format ('txt' or 'pdf')
        
        Returns:
            Path to the saved file
        """
        try:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize company_name and job_title to remove path separators and invalid chars
            safe_company = company_name.replace("/", "_").replace("\\", "_").replace(".", "_")
            safe_job = job_title.replace("/", "_").replace("\\", "_").replace(".", "_")
            
            filename = f"Cover_Letter_Muhammad_Cikal_Merdeka_{safe_company}_{safe_job}".replace(" ", "_")
            
            if format.lower() == "pdf":
                file_path = output_dir / f"{filename}.pdf"
                self._save_as_pdf(cover_letter, file_path, company_name, job_title)
            else:
                file_path = output_dir / f"{filename}.txt"
                self._save_as_txt(cover_letter, file_path)
            
            logger.info(f"Cover letter saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving cover letter: {str(e)}")
            raise
    
    def _save_as_txt(self, content: str, file_path: Path) -> None:
        """Save cover letter as text file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_as_pdf(self, content: str, file_path: Path, company_name: str, 
                     job_title: str) -> None:
        """Save cover letter as PDF file."""
        try:
            # Register Arial font (fallback to default if not available)
            try:
                pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
                font_name = 'Arial'
            except:
                logger.warning("Arial font not found, using Helvetica")
                font_name = 'Helvetica'
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(file_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72,
                title=f"Cover Letter Muhammad Cikal Merdeka - {company_name} - {job_title}",
                author="Muhammad Cikal Merdeka",
                subject="Job Application Cover Letter",
            )
            
            # Define paragraph style
            style = ParagraphStyle(
                'CustomStyle',
                fontName=font_name,
                fontSize=11,
                leading=14.15,
                alignment=TA_JUSTIFY,
                spaceAfter=14.15
            )
            
            # Create story from paragraphs
            story = []
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    p = Paragraph(para.replace('\n', ' '), style)
                    story.append(p)
            
            # Build PDF
            doc.build(story)
            
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            raise
    
    def generate_cold_message(self, job_description: str, company_name: str, 
                             job_title: str, contact_name: str, contact_position: str,
                             resume_link: str) -> str:
        """
        Generate a concise cold message using hybrid context approach.
        
        Args:
            job_description: The job description text
            company_name: Name of the company
            job_title: Title of the position
            contact_name: Name of the contact person
            contact_position: Position/title of the contact person
            resume_link: Link to the resume
        
        Returns:
            Generated cold message text
        """
        try:
            logger.info(f"Generating cold message for {contact_name} ({contact_position}) at {company_name}")
            
            # Get the prompt template with pre-filled candidate info
            template = get_cold_message_prompt(
                candidate_name=CANDIDATE_NAME,
                resume_link=resume_link,
                github_link=GITHUB_LINK,
                website_link=WEBSITE_LINK
            )
            prompt = ChatPromptTemplate.from_template(template)
            
            # Build hybrid context
            context = self._build_context(job_description)
            
            # Generate the cold message
            messages = prompt.format_messages(
                context=context,
                job_description=job_description,
                company_name=company_name,
                job_title=job_title,
                contact_name=contact_name,
                contact_position=contact_position
            )
            
            result = self.llm.invoke(messages)
            logger.info("Cold message generated successfully")
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating cold message: {str(e)}")
            raise
    
    def save_cold_message(self, cold_message: str, contact_name: str, company_name: str) -> str:
        """
        Save the generated cold message to a text file.
        
        Args:
            cold_message: The generated cold message text
            contact_name: Name of the contact person
            company_name: Name of the company
        
        Returns:
            Path to the saved file
        """
        try:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize inputs
            safe_contact = contact_name.replace("/", "_").replace("\\", "_").replace(".", "_")
            safe_company = company_name.replace("/", "_").replace("\\", "_").replace(".", "_")
            
            filename = f"Cold_Message_{safe_contact}_{safe_company}".replace(" ", "_")
            file_path = output_dir / f"{filename}.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cold_message)
            
            logger.info(f"Cold message saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving cold message: {str(e)}")
            raise
