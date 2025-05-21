RAG System
A Retrieval-Augmented Generation (RAG) system for processing text, SQL, Excel, PDF, and image files. The system indexes file content using FAISS and generates responses using an LLM (Azure OpenAI or Vertex AI).
Architecture

File Processors: Extract metadata and content (txt, sql, excel, pdf, image).
FAISS Manager: Indexes embeddings for search.
AIRagModel: Classifies query intent and generates responses.
Oracle DB Manager: Stores metadata (optional).
RagOrchestrator: Orchestrates file processing and query execution.
Catalog: Defines files in config/catalog.yaml.

Setup

Install Dependencies:
pip install -r requirements.txt

Install Tesseract OCR:

Ubuntu: sudo apt-get install tesseract-ocr
macOS: brew install tesseract
Windows: Install from Tesseract releases.


Configure Environment:

For Azure OpenAI, set AZURE_OPENAI_API_KEY or update config/config.yaml:llm:
  provider: "azure"
  azure:
    api_key: "your_azure_api_key"
    endpoint: "https://your-resource-name.openai.azure.com/"
    deployment: "gpt-4o"
embedding:
  provider: "azure"
  azure:
    api_key: "your_azure_api_key"
    endpoint: "https://your-resource-name.openai.azure.com/"
    deployment: "text-embedding-ada-002"


For Vertex AI, set GOOGLE_APPLICATION_CREDENTIALS and update config/config.yaml:llm:
  provider: "vertex"
  vertex:
    project_id: "your_project_id"
    location: "us-central1"
    endpoint: "us-central1-aiplatform.googleapis.com"
    model: "gemini-pro"
embedding:
  provider: "vertex"
  vertex:
    project_id: "your_project_id"
    location: "us-central1"
    endpoint: "us-central1-aiplatform.googleapis.com"
    model: "textembedding-gecko"


Configure LLM hyperparameters:llm:
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9




Prepare Data:

Files in data/ as per config/catalog.yaml:
project_beta/notes/sales_notes.txt: Q1 2024 sales notes.
project_alpha/hr/hr_ddl.sql: HR schema DDL.
project_beta/reports/q1_2024_sales.xlsx: Excel with Metadata and Rules sheets.
project_beta/reports/balance_sheet.pdf: Q1 2024 balance sheet.
project_alpha/images/logo.png: Logo with "Project Alpha" text.




Run:
python main.py



Example Queries
The system runs predefined queries:

"Find all employees in HR schema" (project_alpha)
"Calculate total sales for Q1 2024" (project_beta)
"Summarize the project_beta balance sheet" (project_beta)
"Describe the project_alpha logo" (project_alpha)
"Give me all hard stop rules" (project_beta)

Testing
Run unit tests:
pytest tests/

Development

Add File Type: Create processor in src/file_processor/ and register in processor_factory.py.
Switch Providers: Update llm.provider and embedding.provider in config/config.yaml to switch between Azure OpenAI and Vertex AI.
LLM Hyperparameters: Adjust temperature, max_tokens, top_p in config/config.yaml.
Logging: Configured via logging module.
Sample Files: Located in data/ for testing.

