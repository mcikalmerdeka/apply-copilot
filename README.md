# ApplyCopilot - Your AI Job Application Assistant

An intelligent tool that helps you throughout your job application process. ApplyCopilot creates tailored cover letters for specific job openings and helps you answer employer questions based on your resume, powered by AI.

## 🚀 Features

### 📝 Cover Letter Generator

- **Dual Resume Support**: Supports both AI Engineer and Data-related role resumes
- **Resume Analysis**: Automatically extracts relevant skills and experiences using FAISS vector search
- **Style Matching**: Learns from multiple cover letter examples to maintain consistent writing style
- **Job-Specific Customization**: Tailors content to match specific job requirements
- **PDF & Text Export**: Creates professional documents in multiple formats

### 💬 Employer Q&A Assistant

- **Interactive Chatbot**: Answer employer and recruiter questions based on your indexed resume and job context
- **Context-Aware Responses**: Uses RAG (Retrieval-Augmented Generation) to provide accurate answers from your resume
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

- **Setup-First Workflow**: Centralized Setup section for selecting resume type and entering job details (shared across all features)
- **Tabbed Features**: Easy switching between Cover Letter Generator and Employer Q&A modes
- **Shared State**: Resume and job details persist across tabs - no need to re-enter information
- **Web-Based**: User-friendly Gradio interface accessible in your browser
- **Real-time Feedback**: Status messages and progress indicators

## 🛠️ Implementation Details

- Built with **LangChain** for document processing and LLM integration
- Uses **FAISS** for efficient vector storage and semantic search
- Leverages **OpenAI Embeddings** for text analysis (text-embedding-3-small)
- **GPT-4o Mini** for cover letter generation and chat responses
- **Gradio** web interface with tabbed navigation
- Implements **ReportLab** for PDF document generation
- **Centralized Logging** for application monitoring
- Modular architecture with clean separation of concerns

## 🧩 How It Works

### Cover Letter Generation

1. **Document Processing**:
   - Loads your resume from PDF
   - Splits content into manageable chunks
   - Creates vector embeddings for semantic search

2. **Style Learning**:
   - Analyzes your existing cover letters for tone and format
   - Extracts structural elements and writing patterns

3. **Content Generation**:
   - Retrieves relevant resume sections based on job description
   - Generates tailored content using LLM (GPT-4.1 Mini)
   - Maintains your personal writing style and format

4. **Document Creation**:
   - Formats content with professional styling
   - Outputs a ready-to-submit PDF or TXT document

### Employer Q&A Assistant

1. **Resume Indexing**:
    - Indexes your resume into a searchable vector store
    - Creates semantic embeddings for efficient retrieval
    - Job details are saved separately to provide context

2. **Question Answering**:
    - Retrieves relevant context from your resume based on the question
    - Considers job context (company name, position) to tailor responses
    - Generates professional responses using the retrieved information
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
    - Retrieves relevant skills and experiences from your resume
    - Creates a concise 150-200 word message highlighting your fit for the role
    - Automatically includes your resume link (AI or Data version based on selection)
    - Adds GitHub and personal website links at the end

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
git clone https://github.com/mcikalmerdeka/ApplyCopilot
cd ApplyCopilot
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
   - Resumes: `data/resumes/` (ai_engineer_resume.pdf, data_related_resume.pdf)
   - Cover letter examples: `data/cover_letter_examples/` (example PDFs)
6. **(Optional)** Set your name in `src/config/settings.py`:
   - Edit `CANDIDATE_NAME = "Muhammad Cikal Merdeka"` to your full name for proper signature in generated cover letters

## 🚀 Usage

### Starting the Application

Simply run the main application file:

```bash
uv run app.py
```

This will launch the Gradio web interface at `http://127.0.0.1:7860`

### Using the Web Interface

#### 🔧 Step 1: Setup (Complete This First)

Before using any features, complete the Setup section at the top:

1. **Select Resume Type**: Choose between "AI Engineer" or "Data Related" resume
2. **Index Resume**: Click "📁 Index Resume" to create/load the vector store
3. **Enter Job Details** (shared for all features):
   - Company Name
   - Job Title
   - Full Job Description
4. **Save Job Details**: Click "💾 Save Job Details" to make them available to both features

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
from src.core.vector_store import VectorStoreManager

# Initialize components
generator = CoverLetterGenerator()
vector_store = VectorStoreManager()

# Load and index resume
vector_store.load_and_index_resume("data/resumes/data_related_resume.pdf")
vector_store.save_vector_store("data/vector_stores/data_related")

# Assign to generator
generator.vector_store_manager = vector_store

# Load examples
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
from src.core.vector_store import VectorStoreManager

# Initialize components
generator = CoverLetterGenerator()
vector_store = VectorStoreManager()

# Load and index resume
vector_store.load_and_index_resume("data/resumes/ai_engineer_resume.pdf")
vector_store.save_vector_store("data/vector_stores/ai_engineer")

# Assign to generator
generator.vector_store_manager = vector_store

# Generate cold message
cold_message = generator.generate_cold_message(
    job_description="Your job description here...",
    company_name="Company Name",
    job_title="Position Title",
    contact_name="John Smith",
    contact_position="Tech Lead",
    resume_type="AI Engineer"
)

# Save output
file_path = generator.save_cold_message(
    cold_message,
    contact_name="John Smith",
    company_name="Company Name"
)
```

```python
from src.core.chatbot import EmployerQAChatbot
from src.core.vector_store import VectorStoreManager

# Initialize components
vector_store = VectorStoreManager()
vector_store.load_and_index_resume("data/resumes/ai_engineer_resume.pdf")

chatbot = EmployerQAChatbot(vector_store)

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
│   │   ├── settings.py            # Application configuration (paths, model, CANDIDATE_NAME)
│   │   ├── logging_config.py      # Centralized logging setup
│   │   └── prompts.py             # LLM prompt templates (cover letter + Q&A)
│   ├── core/
│   │   ├── generator.py           # Cover letter and cold message generation logic
│   │   ├── chatbot.py             # Employer Q&A chatbot handler
│   │   └── vector_store.py        # FAISS vector store management
│   └── ui/
│       └── gradio_interface.py    # Gradio web interface with tabs
├── data/
│   ├── resumes/                   # Resume PDF files
│   ├── cover_letter_examples/     # Example cover letters
│   ├── vector_stores/             # FAISS indices (auto-generated)
│   └── output/                    # Generated cover letters
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

## ⚠️ Considerations

- Ensure your `.env` file with API keys is not committed to version control
- The quality of generated content depends on:
  - The completeness of your resume
  - The quality of your existing cover letter templates
  - The specificity of the job description
- The chatbot can only answer questions based on information present in your indexed resume

## 🔒 Privacy

- All processing happens through API calls to OpenAI
- Your resume and cover letter data will be sent to OpenAI services
- Generated content and chat conversations are stored locally
- No data is retained on external servers beyond the API calls

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
