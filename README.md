# Multi-Format Document Chat 📚

A powerful Streamlit-based application that enables interactive conversations with multiple document formats using LangChain and local LLM integration. This application allows users to upload various document types and engage in context-aware conversations about their content.

## Features 🌟

- **Multi-Format Support**: Process various document formats including:
  - PDF (.pdf)
  - Text files (.txt)
  - CSV files (.csv)
  - Excel files (.xlsx, .xls)
  - Word documents (.docx, .doc)

- **Advanced Document Processing**:
  - Automatic text splitting with smart chunk overlap
  - Vector embeddings using HuggingFace's sentence transformers
  - Local vector storage using FAISS

- **Interactive Chat Interface**:
  - Context-aware conversations
  - Source document attribution
  - Chat history management
  - Relevancy scoring for responses

- **User-Friendly Features**:
  - Real-time document processing
  - Source verification
  - Response relevancy calculation
  - Clean and intuitive UI

## Installation 🛠️

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and download the required model:
```bash
# Install Ollama (instructions vary by OS)
ollama pull llama3.2
```

## Required Dependencies 📦

- streamlit
- langchain
- langchain-community
- langchain-core
- sentence-transformers
- faiss-cpu
- ollama
- pandas
- python-docx
- pypdf
- openpyxl

## Usage 🚀

1. Start the application:
```bash
streamlit run app.py
```

2. Upload documents through the sidebar interface

3. Click "Process Documents" to initialize the vector database

4. Start chatting with your documents!

## Features in Detail 🔍

### Document Processing
- Automatic text chunking for optimal processing
- Smart overlap for context preservation
- Metadata preservation for source attribution

### Chat Interface
- Real-time response generation
- Context-aware conversations
- Chat history tracking
- Source document verification

### Quality Assurance
- Relevancy scoring using cosine similarity
- Source document display
- Response verification capabilities

## Architecture 🏗️

The application is built using several key components:

1. **Document Processing Pipeline**:
   - File upload handling
   - Text extraction
   - Chunking
   - Vector embedding

2. **Vector Database**:
   - FAISS for efficient similarity search
   - Local storage for quick access

3. **Conversation Chain**:
   - Context-aware retrieval
   - History-aware processing
   - Response generation

4. **User Interface**:
   - Streamlit-based frontend
   - Responsive chat interface
   - Document management sidebar

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

[Add your chosen license here]

## Support 💬

For support, please [create an issue](your-repository-url/issues) in the repository.

## Acknowledgments 🙏

- LangChain for the excellent RAG framework
- Streamlit for the intuitive UI framework
- HuggingFace for transformer models
- Ollama for local LLM support

## Usage 🚀

Start the application:

streamlit run main.py

1. Upload documents through the sidebar interface
2. Click "Process Documents" to initialize the vector database
3. Start chatting with your documents!

## Demo 🎥

Ollama-Llama3.2-RAG-Chatbot-DemoVideo.mp4

---

Built with ❤️ using Python, Streamlit, and LangChain
