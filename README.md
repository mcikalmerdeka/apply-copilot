---
title: Apply Copilot
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
python_version: "3.12"
app_file: app.py
pinned: false
---

# ApplyCopilot - Your AI Job Application Assistant

<p align="center">
  <img src="assets/Apply Copilot.png" alt="Apply Copilot Logo" height="180" />
</p>

An intelligent tool that helps you throughout your job application process. ApplyCopilot creates tailored cover letters for specific job openings and helps you answer employer questions based on your resume, powered by AI.

Try the application here: [ApplyCopilot HF Spaces](https://huggingface.co/spaces/mcikalmerdeka/apply-copilot)

## 🚀 Features

### 📝 Cover Letter Generator

- **Hybrid Context System**: Uses direct context injection for resumes and RAG (Retrieval-Augmented Generation) for portfolio projects
- **Portfolio Integration**: Optional portfolio upload to showcase relevant projects with semantic search
- **Style Matching**: Learns from multiple cover letter examples to maintain consistent writing style
- **Job-Specific Customization**: Tailors content to match specific job requirements
- **PDF & Text Export**: Creates professional documents in multiple formats

### 💬 Employer Q&A Assistant

- **Interactive Chatbot**: Answer employer and recruiter questions based on your resume and portfolio context
- **Context-Aware Responses**: Combines full resume text with relevant portfolio projects retrieved via semantic search
- **Job-Specific Tailoring**: When job details are saved, responses are tailored to the specific position and company
- **Professional Tone**: Maintains a professional and helpful tone in all responses
- **Practice Mode**: Includes example questions to help you prepare for interviews
- **Persistent Chat**: Clear and restart conversations as needed

### 📨 Cold Message Generator

- **LinkedIn/Email Outreach**: Create concise, professional cold messages to reach out to hiring managers or technical contacts
- **Contact-Specific**: Enter the contact person's name and position for personalized messaging
- **Auto-Link Integration**: Automatically includes your resume, GitHub, and personal website links
- **Short & Impactful**: Generates 150-200 word messages optimized for busy professionals
- **TXT Export**: Save messages for easy copy-paste into LinkedIn, email, or other platforms

### 🎨 User Interface

- **Setup-First Workflow**: Centralized Setup section for uploading documents and entering job details (shared across all features)
- **Optional Portfolio Upload**: Upload a TXT file with your portfolio/projects to enhance context
- **Tabbed Features**: Easy switching between Cover Letter Generator, Employer Q&A, and Cold Message modes
- **Shared State**: Resume, portfolio, and job details persist across tabs - no need to re-enter information
- **Web-Based**: User-friendly Gradio interface accessible in your browser
- **Real-time Feedback**: Status messages and progress indicators

## 🛠️ Implementation Details

- Built with **LangChain** for document processing and LLM integration
- **Hybrid Context Approach**: Direct injection for resumes (full text) + RAG for portfolios (semantic search)
- Uses **FAISS** for efficient vector storage of portfolio projects
- Leverages **OpenAI Embeddings** for portfolio semantic search (text-embedding-3-small)
- **Claude Sonnet 4.6** via Anthropic API for cover letter generation and chat responses
- **Gradio** web interface with tabbed navigation
- Implements **ReportLab** for PDF document generation
- **Centralized Logging** for application monitoring
- Modular architecture with clean separation of concerns

## 🧩 How It Works

### Cover Letter Generation

1. **Document Processing**:
   - Loads your resume from PDF (uses full text - direct injection)
   - Optionally loads your portfolio from TXT (uses semantic search - RAG)
   - Analyzes cover letter examples for style reference

2. **Hybrid Context Assembly**:
   - **Resume**: Injected directly into context as full text for complete work history
   - **Portfolio**: Relevant projects retrieved via semantic search to match job requirements
   - This hybrid approach ensures complete resume coverage while highlighting most relevant projects

3. **Style Learning**:
   - Analyzes your existing cover letters for tone and format
   - Extracts structural elements and writing patterns

4. **Content Generation**:
   - Combines resume text with relevant portfolio projects based on job description
   - Generates tailored content using Claude Sonnet 4.6
   - References specific portfolio projects that demonstrate relevant experience
   - Maintains your personal writing style and format

5. **Document Creation**:
   - Formats content with professional styling
   - Outputs a ready-to-submit PDF or TXT document

### Employer Q&A Assistant

1. **Context Preparation**:
   - Resume text stored for direct injection (no RAG - complete context)
   - Portfolio indexed with semantic embeddings for RAG retrieval (when uploaded)
   - Job details saved separately to provide context

2. **Question Answering**:
   - Uses full resume text as primary context for complete information
   - Retrieves relevant portfolio projects via semantic search when applicable
   - Considers job context (company name, position) to tailor responses
   - Generates professional responses using Claude Sonnet 4.6
   - When job details are available, emphasizes fit for the specific role
   - Maintains conversation history for context-aware responses

3. **Interactive Chat**:
   - Real-time chat interface with employers/recruiters
   - Example questions to practice your responses
   - Easy conversation management (clear, save)

### Cold Message Generator

1. **Contact Information**:
   - Enter the contact person's name and position (e.g., HR Manager, Tech Lead)
   - The position helps tailor the message tone (formal for HR, technical for engineers)

2. **Message Generation**:
   - Uses full resume context combined with relevant portfolio projects
   - Creates a concise 150-200 word message highlighting your fit for the role
   - Automatically includes your resume, GitHub, and personal website links
   - Follows proven templates for AI Engineering and Data roles

3. **Professional Formatting**:
   - Generates a friendly yet professional greeting
   - Clearly states your interest in the position
   - Highlights 2-3 relevant skills that match the job requirements
   - Includes a clear call-to-action for next steps
   - Formatted for easy copy-paste into LinkedIn messages or emails

## 📋 Requirements

- Python 3.12+
- OpenAI API key for embeddings and generation
- Required Python packages (see requirements.txt)

## 🔧 Setup

1. Clone this repository:

```bash
git clone https://github.com/mcikalmerdeka/apply-copilot
cd apply-copilot
```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   # Or using uv:
   uv add -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```
5. Ensure your documents are in place:
   - Cover letter examples: `data/cover_letter_examples/` (example PDFs)
6. **(Optional)** Customize your settings in `src/config/settings.py`:
   - Edit `CANDIDATE_NAME = "Muhammad Cikal Merdeka"` to your full name for proper signature
   - Update `RESUME_AI_LINK`, `RESUME_DATA_LINK`, `GITHUB_LINK`, `WEBSITE_LINK` with your links

## 🚀 Usage

### Starting the Application

Simply run the main application file:

```bash
uv run app.py
```

This will launch the Gradio web interface at `http://127.0.0.1:7860`

### Using the Web Interface

#### 🔧 Setup (Complete This First)

Before using any features, complete the Setup section at the top:

1. **Upload Resume (Required)**: Upload your resume PDF file
   - Click "📁 Index Resume" to process the resume
   - Uses direct context injection (full resume text)

2. **Upload Portfolio (Optional)**: Upload a TXT file with your portfolio/projects
   - Click "📁 Index Portfolio" to process the portfolio
   - Uses RAG for semantic retrieval of relevant projects
   - Include project descriptions, technologies used, and outcomes

3. **Restart if needed**: Click "🔄 Restart Application" to clear all data and start fresh

4. **Enter Job Details** (shared for all features):
   - Company Name
   - Job Title
   - Full Job Description
5. **Save Job Details**: Click "💾 Save Job Details" to make them available to all features

#### Tab 1: 📝 Cover Letter Generator

After completing Setup:

1. **Choose Output Format**: Select TXT or PDF
2. **Generate**: Click "✨ Generate Cover Letter" to create your personalized cover letter
3. **Download**: Download the generated cover letter from the output section

#### Tab 2: 💬 Employer Q&A Assistant

After completing Setup:

1. **Verify Context**: Check the "Current Context" status to confirm resume and job details are loaded
2. **Start Chatting**:
   - Type employer questions in the chat box
   - Or click on example questions to practice
   - The AI will answer based on your resume content, tailored to the specific job
3. **Manage Conversations**:
   - Use "🗑️ Clear Chat" to start a new conversation
   - Review your chat history for reference

#### Tab 3: 📨 Message to Contact

After completing Setup:

1. **Enter Contact Details**:
   - Contact Name: The person you're reaching out to (e.g., "Sarah Johnson")
   - Contact Position: Their role (e.g., "HR Manager", "Tech Lead", "Engineering Manager")
2. **Generate Message**: Click "✨ Generate Cold Message" to create your outreach message
3. **Use the Message**:
   - Copy the generated message directly from the output
   - Or download the TXT file
   - Paste into LinkedIn, email, or other messaging platforms

### Example Use Cases

**Cover Letter Generation**:

```python
from src.core.generator import CoverLetterGenerator

# Initialize generator
generator = CoverLetterGenerator()

# Load resume (direct injection - full text)
generator.vector_store_manager.load_and_index_resume("path/to/your/resume.pdf")

# Load portfolio (optional - RAG for project retrieval)
generator.load_portfolio("path/to/your/portfolio.txt")

# Load cover letter examples for style matching
generator.load_cover_letter_examples()

# Generate cover letter
cover_letter = generator.generate_cover_letter(
    job_description="Your job description here...",
    company_name="Company Name",
    job_title="Position Title"
)

# Save output
file_path = generator.save_cover_letter(
    cover_letter,
    company_name="Company Name",
    job_title="Position Title",
    format="pdf"  # or "txt"
)
```

**Cold Message Generator**:

```python
from src.core.generator import CoverLetterGenerator

# Initialize generator (automatically loads links from settings)
generator = CoverLetterGenerator()

# Load resume
generator.vector_store_manager.load_and_index_resume("path/to/your/resume.pdf")

# Optionally load portfolio
generator.load_portfolio("path/to/your/portfolio.txt")

# Generate cold message
cold_message = generator.generate_cold_message(
    job_description="Your job description here...",
    company_name="Company Name",
    job_title="Position Title",
    contact_name="John Smith",
    contact_position="Tech Lead",
    resume_link="https://drive.google.com/your-resume-link"
)

# Save output
file_path = generator.save_cold_message(
    cold_message,
    contact_name="John Smith",
    company_name="Company Name"
)
```

**Employer Q&A Chatbot**:

```python
from src.core.chatbot import EmployerQAChatbot
from src.core.generator import CoverLetterGenerator

# Initialize via generator for shared context
generator = CoverLetterGenerator()
generator.vector_store_manager.load_and_index_resume("path/to/your/resume.pdf")
generator.load_portfolio("path/to/your/portfolio.txt")

chatbot = EmployerQAChatbot(generator.vector_store_manager)

# Answer employer questions
response = chatbot.chat(
    message="What is your experience with Python?",
    history=[]  # Previous conversation history
)
print(response)
```

## 📊 Project Structure

```
├── app.py                         # Main application entry point
├── src/
│   ├── config/
│   │   ├── settings.py            # Application configuration (paths, model, CANDIDATE_NAME, links)
│   │   ├── logging_config.py      # Centralized logging setup
│   │   └── prompts.py             # LLM prompt templates
│   ├── core/
│   │   ├── generator.py           # Cover letter and cold message generation logic
│   │   ├── chatbot.py             # Employer Q&A chatbot handler
│   │   └── vector_store.py        # Hybrid: direct injection (resume) + RAG (portfolio)
│   └── ui/
│       └── gradio_interface.py    # Gradio web interface with tabs
├── data/
│   ├── resumes/                   # Uploaded resume PDF files
│   ├── cover_letter_examples/     # Example cover letters for style matching
│   ├── vector_stores/             # FAISS indices for portfolio (auto-generated)
│   └── output/                    # Generated cover letters and cold messages
├── assets/                        # Static assets (logos, images)
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

## ⚠️ Considerations

- Ensure your `.env` file with API keys is not committed to version control
- The quality of generated content depends on:
  - The completeness of your resume and portfolio
  - The quality of your existing cover letter templates
  - The specificity of the job description
- **Resume vs Portfolio Approach**:
  - Resume uses direct context injection (complete text) for accurate representation
  - Portfolio uses RAG (semantic search) to highlight most relevant projects
  - Without a portfolio, the system works with resume only
- Portfolio file should be a plain text (.txt) file with project descriptions

## 🔒 Privacy

- All processing happens through API calls to OpenAI
- Your resume and cover letter data will be sent to OpenAI services
- Generated content and chat conversations are stored locally
- No data is retained on external servers beyond the API calls

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
