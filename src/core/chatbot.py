"""
Employer Q&A Chatbot handler for ApplyCopilot.
Handles conversations with employers/recruiters based on indexed resume and portfolio.
Uses hybrid approach: Resume (direct injection) + Portfolio (RAG)
"""

from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config.logging_config import setup_logger
from src.config.settings import LLM_MODEL, CANDIDATE_NAME
from src.config.prompts import get_employer_qa_system_prompt

logger = setup_logger(__name__)


class EmployerQAChatbot:
    """Chatbot for answering employer questions based on resume (direct) and portfolio (RAG)."""
    
    def __init__(self, vector_store_manager, llm_model: str = LLM_MODEL):
        """
        Initialize the employer Q&A chatbot.
        
        Args:
            vector_store_manager: VectorStoreManager instance with loaded resume/portfolio
            llm_model: LLM model name to use
        """
        self.llm = ChatAnthropic(model=llm_model, temperature=0.7)
        self.vector_store_manager = vector_store_manager
        self.chat_history: List[Dict[str, str]] = []
        self.candidate_name = CANDIDATE_NAME
        
        # Job context (optional, for more contextual answers)
        self.job_context: Optional[str] = None
        self.job_description: Optional[str] = None
        
        logger.info(f"Initialized EmployerQAChatbot with LLM model: {llm_model}")
    
    def set_job_context(self, job_context: str, job_description: str = "") -> None:
        """
        Set job context for more contextual answers.
        
        Args:
            job_context: String describing the position (e.g., "ML Engineer at DOHE AI")
            job_description: Full job description text
        """
        self.job_context = job_context
        self.job_description = job_description
        logger.info(f"Job context set: {job_context}")
    
    def clear_job_context(self) -> None:
        """Clear the job context."""
        self.job_context = None
        self.job_description = None
        logger.info("Job context cleared")
    
    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history."""
        return self.chat_history.copy()
    
    def _build_context(self, question: str) -> str:
        """
        Build context using hybrid approach:
        - Resume: Direct injection (full text, no RAG)
        - Portfolio: RAG retrieval (if available)
        
        Args:
            question: The employer's question for portfolio retrieval
            
        Returns:
            Combined context string
        """
        context_parts = []
        
        # 1. Add resume context (always direct injection)
        if self.vector_store_manager.has_resume():
            resume_context = self.vector_store_manager.get_resume_context(use_rag=False)
            context_parts.append("=== RESUME ===\n" + resume_context)
            logger.info(f"Added resume context to chat ({len(resume_context)} chars)")
        
        # 2. Add portfolio context via RAG (if available)
        if self.vector_store_manager.has_portfolio():
            portfolio_context = self.vector_store_manager.get_portfolio_context(question)
            if portfolio_context:
                context_parts.append("\n\n=== RELEVANT PROJECTS FROM PORTFOLIO ===\n" + portfolio_context)
                logger.info(f"Added portfolio context to chat via RAG ({len(portfolio_context)} chars)")
        
        return "\n\n".join(context_parts) if context_parts else "No context available."
    
    def answer_question(self, question: str, history: List[Dict[str, Any]]) -> str:
        """
        Answer an employer question based on hybrid context (resume direct + portfolio RAG).
        
        Args:
            question: The employer's question
            history: List of previous messages in OpenAI format
        
        Returns:
            Generated response
        """
        try:
            logger.info(f"Processing employer question: {question[:100]}...")
            
            # Check if resume is available
            if not self.vector_store_manager.has_resume():
                return "❌ Error: No resume indexed yet. Please index a resume first before using the chatbot."
            
            # Build hybrid context (resume direct + portfolio RAG)
            context = self._build_context(question)
            
            # Get system prompt with optional job context
            system_prompt = get_employer_qa_system_prompt(
                job_context=self.job_context,
                job_description=self.job_description
            )
            
            # Prepare messages for the LLM
            messages = []
            
            # Add system context
            messages.append(SystemMessage(content=system_prompt))
            
            # Add previous chat history
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            
            # Add current question with hybrid context
            current_prompt = f"""**Candidate Context (Resume + Portfolio):**
{context}

**Employer's Question:**
{question}

**Context Usage Guidelines:**
- Use the RESUME section for work experience, education, and core skills
- Use the PORTFOLIO section for specific project examples and technical demonstrations
- Reference specific projects from the portfolio when relevant to the question
- Answer based ONLY on the information available in the context above

**Your Response:**
Please provide a helpful, professional answer to the employer's question based on the candidate context above."""
            messages.append(HumanMessage(content=current_prompt))
            
            # Generate response
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info(f"Generated answer (length: {len(answer)} chars)")
            return answer
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def chat(self, message: str, history: List[Dict[str, Any]]) -> str:
        """
        Main chat interface function for Gradio ChatInterface.
        
        Args:
            message: User's message
            history: Chat history in Gradio format
        
        Returns:
            Assistant's response
        """
        try:
            # Convert Gradio history format to our format if needed
            formatted_history = []
            if history:
                for msg in history:
                    if isinstance(msg, dict):
                        formatted_history.append(msg)
                    elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                        # Handle old format [user_msg, assistant_msg]
                        if msg[0]:
                            formatted_history.append({"role": "user", "content": msg[0]})
                        if msg[1]:
                            formatted_history.append({"role": "assistant", "content": msg[1]})
            
            response = self.answer_question(message, formatted_history)
            return response
            
        except Exception as e:
            error_msg = f"❌ Error in chat: {str(e)}"
            logger.error(error_msg)
            return error_msg
