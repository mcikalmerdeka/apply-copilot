import os
import shutil
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

from src.config.logging_config import setup_logger
from src.config.settings import RESUMES_DIR, VECTOR_STORES_DIR, CANDIDATE_NAME, DATA_DIR
from src.core.generator import CoverLetterGenerator
from src.core.chatbot import EmployerQAChatbot

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)


class ApplyCopilotUI:
    """Gradio UI for ApplyCopilot - Cover Letter Generation and Employer Q&A."""
    
    def __init__(self):
        """Initialize the UI and components."""
        self.generator = CoverLetterGenerator()
        self.chatbot = EmployerQAChatbot(self.generator.vector_store_manager)
        
        # Shared state
        self.current_resume_path = None
        self.current_portfolio_path = None
        self.job_details = {
            "company_name": "",
            "job_title": "",
            "job_description": ""
        }
        
        logger.info("Initialized ApplyCopilotUI")
    
    def index_resume(self, resume_file) -> str:
        """
        Index a resume from uploaded file using direct injection (no RAG).
        
        Args:
            resume_file: The uploaded resume file (Gradio file object)
        
        Returns:
            Status message
        """
        try:
            if resume_file is None:
                return "❌ Please upload a resume file first."
            
            # Handle Gradio file upload - resume_file is a file path or dict
            if isinstance(resume_file, dict):
                uploaded_path = resume_file.get('name')
            elif isinstance(resume_file, str):
                uploaded_path = resume_file
            else:
                return f"❌ Invalid file format. Please upload a PDF file."
            
            if not uploaded_path or not os.path.exists(uploaded_path):
                return "❌ Error: Could not access uploaded file."
            
            # Copy uploaded file to resumes directory with timestamp to avoid conflicts
            import time
            timestamp = int(time.time())
            resume_filename = f"uploaded_resume_{timestamp}.pdf"
            resume_path = Path(RESUMES_DIR) / resume_filename
            
            # Copy the uploaded file to our data directory
            shutil.copy2(uploaded_path, resume_path)
            logger.info(f"Saved uploaded resume to: {resume_path}")
            
            # Always create fresh vector store - delete any existing temp stores
            temp_vector_path = Path(VECTOR_STORES_DIR) / f"temp_{timestamp}"
            if temp_vector_path.exists():
                shutil.rmtree(temp_vector_path)
            
            # Load and index the resume (direct injection approach)
            logger.info(f"Indexing resume with direct injection approach")
            result = self.generator.vector_store_manager.load_and_index_resume(str(resume_path))
            self.generator.vector_store_manager.save_vector_store(str(temp_vector_path), store_type="resume")
            
            # Store the current resume path
            self.current_resume_path = str(resume_path)
            
            # Load cover letter examples
            self.generator.load_cover_letter_examples()
            
            message = f"✅ Resume indexed successfully: {Path(uploaded_path).name} ({result['text_length']} chars) - Using direct context injection"
            logger.info(message)
            return message
            
        except Exception as e:
            error_msg = f"❌ Error indexing resume: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def index_portfolio(self, portfolio_file) -> str:
        """
        Index a portfolio from uploaded file using RAG approach.
        
        Args:
            portfolio_file: The uploaded portfolio file (Gradio file object)
        
        Returns:
            Status message
        """
        try:
            if portfolio_file is None:
                return "⚠️ No portfolio file uploaded. This is optional - you can proceed without a portfolio."
            
            # Handle Gradio file upload
            if isinstance(portfolio_file, dict):
                uploaded_path = portfolio_file.get('name')
            elif isinstance(portfolio_file, str):
                uploaded_path = portfolio_file
            else:
                return f"❌ Invalid file format. Please upload a TXT file."
            
            if not uploaded_path or not os.path.exists(uploaded_path):
                return "❌ Error: Could not access uploaded file."
            
            # Copy uploaded file to data directory
            import time
            timestamp = int(time.time())
            portfolio_filename = f"uploaded_portfolio_{timestamp}.txt"
            portfolio_path = Path(DATA_DIR) / portfolio_filename
            
            # Copy the uploaded file
            shutil.copy2(uploaded_path, portfolio_path)
            logger.info(f"Saved uploaded portfolio to: {portfolio_path}")
            
            # Index portfolio with RAG approach
            logger.info(f"Indexing portfolio with RAG approach")
            result = self.generator.load_portfolio(str(portfolio_path))
            
            # Store the current portfolio path
            self.current_portfolio_path = str(portfolio_path)
            
            message = f"✅ Portfolio indexed successfully: {Path(uploaded_path).name} ({result['text_length']} chars, {result['chunks_created']} chunks) - Using RAG retrieval"
            logger.info(message)
            return message
            
        except Exception as e:
            error_msg = f"❌ Error indexing portfolio: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def restart_application(self) -> tuple:
        """
        Restart the application by clearing all state and deleting temporary files.
        
        Returns:
            Tuple of cleared values for UI components
        """
        try:
            # Clear vector store
            self.generator.vector_store_manager.clear_vector_store(store_type="all")
            
            # Clear chatbot history
            self.chatbot.clear_history()
            self.chatbot.clear_job_context()
            
            # Clear job details
            self.job_details = {
                "company_name": "",
                "job_title": "",
                "job_description": ""
            }
            
            # Clear file paths
            self.current_resume_path = None
            self.current_portfolio_path = None
            
            # Clean up temporary vector stores
            if VECTOR_STORES_DIR.exists():
                for item in VECTOR_STORES_DIR.iterdir():
                    if item.is_dir() and item.name.startswith("temp_"):
                        try:
                            shutil.rmtree(item)
                            logger.info(f"Deleted temporary vector store: {item}")
                        except Exception as e:
                            logger.warning(f"Could not delete {item}: {e}")
            
            # Clean up uploaded resumes
            if RESUMES_DIR.exists():
                for item in RESUMES_DIR.iterdir():
                    if item.is_file() and item.name.startswith("uploaded_resume_"):
                        try:
                            item.unlink()
                            logger.info(f"Deleted uploaded resume: {item}")
                        except Exception as e:
                            logger.warning(f"Could not delete {item}: {e}")
            
            # Clean up uploaded portfolios
            for item in DATA_DIR.iterdir():
                if item.is_file() and item.name.startswith("uploaded_portfolio_"):
                    try:
                        item.unlink()
                        logger.info(f"Deleted uploaded portfolio: {item}")
                    except Exception as e:
                        logger.warning(f"Could not delete {item}: {e}")
            
            logger.info("Application restarted successfully")
            return (
                "🔄 Application restarted. Please upload a new resume and enter job details.",  # index_status
                "No portfolio indexed (optional).",   # portfolio_status
                "No job details saved yet.",   # job_status
                "❌ No context loaded. Please upload a resume and enter job details.",  # qa_context_status
                None,   # resume_upload (clear file)
                None,   # portfolio_upload (clear file)
                "",     # company_name
                "",     # job_title
                "",     # job_description
                []      # chatbot (clear chat history)
            )
            
        except Exception as e:
            error_msg = f"❌ Error restarting application: {str(e)}"
            logger.error(error_msg)
            return (
                error_msg,
                "",
                "",
                "",
                None,
                None,
                "",
                "",
                "",
                []
            )
    
    def update_job_details(self, company_name: str, job_title: str, job_description: str) -> str:
        """
        Update shared job details for both features.
        
        Args:
            company_name: Name of the company
            job_title: Job title/position
            job_description: Full job description
        
        Returns:
            Status message
        """
        try:
            self.job_details["company_name"] = company_name
            self.job_details["job_title"] = job_title
            self.job_details["job_description"] = job_description
            
            # Update chatbot with job context if available
            if company_name or job_title:
                job_context = f"Position: {job_title} at {company_name}" if job_title and company_name else f"Position: {job_title or company_name}"
                self.chatbot.set_job_context(job_context, job_description)
            
            logger.info(f"Updated job details: {job_title} at {company_name}")
            return f"✅ Job details updated: {job_title} at {company_name}"
            
        except Exception as e:
            error_msg = f"❌ Error updating job details: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def generate_cover_letter(self, output_format: str) -> tuple:
        """
        Generate a cover letter using shared job details.
        
        Args:
            output_format: Output format ('txt' or 'pdf')
        
        Returns:
            Tuple of (cover letter text, file path, status message)
        """
        try:
            # Validate inputs
            if not self.job_details["company_name"] or not self.job_details["job_title"] or not self.job_details["job_description"]:
                return "", None, "❌ Please fill in all job details in the Setup section"
            
            if not self.generator.vector_store_manager.has_resume():
                return "", None, "❌ Please upload and index a resume first in the Setup section"
            
            # Generate cover letter
            logger.info(f"Generating cover letter for {self.job_details['company_name']} - {self.job_details['job_title']}")
            cover_letter = self.generator.generate_cover_letter(
                job_description=self.job_details["job_description"],
                company_name=self.job_details["company_name"],
                job_title=self.job_details["job_title"]
            )
            
            # Save cover letter
            file_path = self.generator.save_cover_letter(
                cover_letter=cover_letter,
                company_name=self.job_details["company_name"],
                job_title=self.job_details["job_title"],
                format=output_format
            )
            
            success_msg = f"✅ Cover letter generated and saved to: {file_path}"
            logger.info(success_msg)
            
            return cover_letter, file_path, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error generating cover letter: {str(e)}"
            logger.error(error_msg)
            return "", None, error_msg
    
    def generate_cold_message(self, contact_name: str, contact_position: str, resume_link: str) -> tuple:
        """
        Generate a cold message using shared job details and contact info.
        
        Args:
            contact_name: Name of the contact person
            contact_position: Position/title of the contact person
            resume_link: Link to the resume (Google Drive or other)
        
        Returns:
            Tuple of (cold message text, file path, status message)
        """
        try:
            # Validate inputs
            if not self.job_details["company_name"] or not self.job_details["job_title"] or not self.job_details["job_description"]:
                return "", None, "❌ Please fill in all job details in the Setup section"
            
            if not self.generator.vector_store_manager.has_resume():
                return "", None, "❌ Please upload and index a resume first in the Setup section"
            
            if not contact_name or not contact_position:
                return "", None, "❌ Please enter both contact name and position"
            
            if not resume_link:
                return "", None, "❌ Please provide a link to your resume"
            
            # Generate cold message
            logger.info(f"Generating cold message for {contact_name} ({contact_position}) at {self.job_details['company_name']}")
            cold_message = self.generator.generate_cold_message(
                job_description=self.job_details["job_description"],
                company_name=self.job_details["company_name"],
                job_title=self.job_details["job_title"],
                contact_name=contact_name,
                contact_position=contact_position,
                resume_link=resume_link
            )
            
            # Save cold message
            file_path = self.generator.save_cold_message(
                cold_message=cold_message,
                contact_name=contact_name,
                company_name=self.job_details["company_name"]
            )
            
            success_msg = f"✅ Cold message generated and saved to: {file_path}"
            logger.info(success_msg)
            
            return cold_message, file_path, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error generating cold message: {str(e)}"
            logger.error(error_msg)
            return "", None, error_msg
    
    def create_setup_section(self) -> tuple:
        """Create the shared Setup section with Resume upload, Portfolio upload, and Job Details.
        
        Returns:
            Tuple of UI components for restart handling
        """
        with gr.Row() as setup_row:
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Step 1: Upload Resume (Required)")
                gr.Markdown("*Resume uses direct context injection (full text)*")
                resume_upload = gr.File(
                    label="Upload Resume PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                index_btn = gr.Button("📁 Index Resume", variant="primary")
                index_status = gr.Textbox(
                    label="Resume Status",
                    value="No resume indexed yet. Please upload a resume PDF.",
                    interactive=False
                )
                
                gr.Markdown("### 📂 Step 2: Upload Portfolio (Optional)")
                gr.Markdown("*Portfolio uses RAG for semantic retrieval of relevant projects*")
                portfolio_upload = gr.File(
                    label="Upload Portfolio TXT",
                    file_types=[".txt"],
                    type="filepath"
                )
                portfolio_btn = gr.Button("📁 Index Portfolio", variant="secondary")
                portfolio_status = gr.Textbox(
                    label="Portfolio Status",
                    value="No portfolio indexed (optional). Upload a .txt file to include portfolio projects.",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Step 3: Enter Job Details (Shared for All Features)")
                gr.Markdown("*These details will be used for Cover Letter generation, Employer Q&A, and Cold Message*")
                
                with gr.Row():
                    company_name = gr.Textbox(
                        label="Company Name",
                        placeholder="e.g., PT Example Company",
                        scale=1
                    )
                    job_title = gr.Textbox(
                        label="Job Title",
                        placeholder="e.g., Senior Data Analyst",
                        scale=1
                    )
                
                job_description = gr.Textbox(
                    label="Job Description",
                    placeholder="Paste the full job description here...",
                    lines=5
                )
                
                with gr.Row():
                    update_job_btn = gr.Button("💾 Save Job Details", variant="secondary")
                    restart_btn = gr.Button("🔄 Restart Application", variant="stop")
                
                job_status = gr.Textbox(
                    label="Job Details Status",
                    value="No job details saved yet.",
                    interactive=False
                )
        
        # Event handlers for setup section
        index_btn.click(
            fn=self.index_resume,
            inputs=[resume_upload],
            outputs=[index_status]
        )
        
        portfolio_btn.click(
            fn=self.index_portfolio,
            inputs=[portfolio_upload],
            outputs=[portfolio_status]
        )
        
        update_job_btn.click(
            fn=self.update_job_details,
            inputs=[company_name, job_title, job_description],
            outputs=[job_status]
        )
        
        return (setup_row, resume_upload, index_status, portfolio_upload, portfolio_status,
                job_status, company_name, job_title, job_description, restart_btn)
    
    def create_cover_letter_tab(self) -> gr.Tab:
        """Create the Cover Letter Generation tab."""
        with gr.Tab("📝 Cover Letter Generator", id="cover_letter") as tab:
            gr.Markdown("## Generate tailored cover letters using AI based on your resume and job details.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Generation Settings")
                    output_format = gr.Radio(
                        choices=["txt", "pdf"],
                        label="Output Format",
                        value="txt"
                    )
                    generate_btn = gr.Button("✨ Generate Cover Letter", variant="primary")
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Generated Cover Letter")
                    generation_status = gr.Textbox(label="Generation Status", interactive=False)
                    cover_letter_output = gr.Textbox(
                        label="Generated Cover Letter",
                        lines=15,
                        interactive=False
                    )
                    file_output = gr.File(label="Download Cover Letter")
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_cover_letter,
                inputs=[output_format],
                outputs=[cover_letter_output, file_output, generation_status]
            )
        
        return tab
    
    def create_employer_qa_tab(self) -> gr.Tab:
        """Create the Employer Q&A Chatbot tab."""
        with gr.Tab("💬 Employer Q&A Assistant", id="employer_qa") as tab:
            gr.Markdown(f"## Chat with employers on behalf of {CANDIDATE_NAME}")
            gr.Markdown(
                "Answer questions from recruiters and hiring managers based on your indexed resume and job details. "
                "The AI assistant will provide professional responses using your background information."
            )
            
            # Show current context
            with gr.Row():
                with gr.Column(scale=1):
                    qa_context_status = gr.Textbox(
                        label="Current Context",
                        value="No context loaded. Please upload a resume and enter job details in the Setup section.",
                        interactive=False
                    )
            
            gr.Markdown("---")
            
            # Chat Interface
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 Chat with Employers")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        placeholder="Start a conversation with potential employers..."
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Message",
                            placeholder="Type employer's question here or ask a practice question...",
                            scale=7,
                            show_label=False
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        save_btn = gr.Button("💾 Save Conversation", variant="secondary")
            
            # Chat status
            chat_status = gr.Textbox(label="Chat Status", interactive=False, visible=False)
            
            # Chat examples
            gr.Markdown("### 💡 Example Questions You Might Receive")
            gr.Examples(
                examples=[
                    "Can you tell me about your experience with machine learning?",
                    "What projects have you worked on related to data analysis?",
                    "Are you familiar with Python and its data science libraries?",
                    "What is your experience with cloud platforms like AWS or GCP?",
                    "Can you describe a challenging technical problem you've solved?",
                    "What is your educational background?",
                    "Are you comfortable working in a team environment?",
                    "What are your salary expectations?",
                    "When would you be available to start?",
                ],
                inputs=[msg_input],
                label="Click any example to use it"
            )
            
            # Event handlers
            def respond(message, history):
                if not message.strip():
                    return "", history
                
                if not self.generator.vector_store_manager.has_resume():
                    history.append({"role": "assistant", "content": "❌ Please load a resume first by uploading a PDF in the Setup section."})
                    return "", history
                
                try:
                    response = self.chatbot.chat(message, history)
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response})
                    return "", history
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": error_msg})
                    return "", history
            
            def clear_chat():
                self.chatbot.clear_history()
                return []
            
            def update_qa_context():
                """Update the context status display."""
                if not self.generator.vector_store_manager.has_resume():
                    return "❌ No resume indexed. Please upload a resume PDF in the Setup section."
                
                context_info = "✅ Resume indexed (direct injection)"
                
                if self.generator.vector_store_manager.has_portfolio():
                    context_info += " | Portfolio indexed (RAG)"
                
                if self.job_details.get("job_title"):
                    context_info += f" | Job: {self.job_details['job_title']}"
                if self.job_details.get("company_name"):
                    context_info += f" at {self.job_details['company_name']}"
                
                return context_info
            
            # Update context status when tab is opened
            tab.select(
                fn=update_qa_context,
                outputs=[qa_context_status]
            )
            
            submit_btn.click(
                fn=respond,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            msg_input.submit(
                fn=respond,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot]
            )
        
        return tab, chatbot, qa_context_status
    
    def create_cold_message_tab(self) -> gr.Tab:
        """Create the Cold Message / Outreach tab."""
        with gr.Tab("📨 Message to Contact", id="cold_message") as tab:
            gr.Markdown(f"## Generate a cold message to reach out to hiring managers or technical contacts")
            gr.Markdown(
                "Create a concise, professional cold message or LinkedIn connection request. "
                "The message will include links to your resume, GitHub, and personal website."
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Contact Information")
                    
                    contact_name = gr.Textbox(
                        label="Contact Name",
                        placeholder="e.g., John Doe",
                        info="Name of the person you're reaching out to"
                    )
                    
                    contact_position = gr.Textbox(
                        label="Contact Position",
                        placeholder="e.g., HR Manager, Tech Lead, Engineering Manager",
                        info="Their role (helps tailor the message tone)"
                    )
                    
                    resume_link = gr.Textbox(
                        label="Resume Link",
                        placeholder="e.g., https://drive.google.com/file/d/...",
                        info="Link to your resume (Google Drive or other)"
                    )
                    
                    generate_btn = gr.Button("✨ Generate Cold Message", variant="primary")
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Generated Cold Message")
                    generation_status = gr.Textbox(label="Generation Status", interactive=False)
                    
                    cold_message_output = gr.Textbox(
                        label="Cold Message",
                        lines=10,
                        interactive=False,
                        info="Copy this message to send via LinkedIn, email, or other platforms"
                    )
                    
                    file_output = gr.File(label="Download Message (TXT)")
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_cold_message,
                inputs=[contact_name, contact_position, resume_link],
                outputs=[cold_message_output, file_output, generation_status]
            )
        
        return tab
    
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface with shared setup and tabs."""
        
        with gr.Blocks(title="ApplyCopilot - Your AI Job Application Assistant") as interface:
            gr.Markdown("# 🤖 ApplyCopilot")
            gr.Markdown(
                "**Your intelligent job application assistant with hybrid RAG approach.** "
                "Resume uses direct context injection, Portfolio uses semantic search (RAG)."
            )
            
            # Setup Section (Shared across all features)
            with gr.Column():
                gr.Markdown("## 🔧 Setup")
                gr.Markdown("*Complete these steps. Upload your resume (required) and optionally a portfolio, then enter job details.*")
                (setup_row, resume_upload, index_status, portfolio_upload, portfolio_status,
                 job_status, company_name, job_title, job_description, restart_btn) = self.create_setup_section()
            
            gr.Markdown("---")
            
            # Feature Tabs
            gr.Markdown("## 🚀 Features")
            with gr.Tabs():
                self.create_cover_letter_tab()
                qa_tab, qa_chatbot, qa_context_status = self.create_employer_qa_tab()
                self.create_cold_message_tab()
            
            gr.Markdown("---")
            gr.Markdown(
                f"*Powered by AI • Hybrid RAG Approach • Built for {CANDIDATE_NAME}*"
            )
            
            # Connect restart button to clear all UI components
            restart_btn.click(
                fn=self.restart_application,
                outputs=[
                    index_status,       # Status message
                    portfolio_status,   # Portfolio status
                    job_status,         # Job status
                    qa_context_status,  # QA context status
                    resume_upload,      # Clear file upload
                    portfolio_upload,   # Clear portfolio upload
                    company_name,       # Clear company name
                    job_title,          # Clear job title
                    job_description,    # Clear job description
                    qa_chatbot          # Clear chat history
                ]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(theme=gr.themes.Soft(), **kwargs)


def main():
    """Main entry point for the UI."""
    ui = ApplyCopilotUI()
    ui.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
