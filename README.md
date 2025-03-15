# Document RAG System

A modern web-based Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about their content. The system uses state-of-the-art language models and embeddings to provide accurate, context-aware responses.

## Features

- 🚀 Modern, responsive web interface
- 🌓 Dark/Light mode support
- 📁 Multiple document format support (PDF, DOC, DOCX, TXT)
- 🔄 Drag and drop file upload
- 💬 Interactive chat-like interface
- 🤖 Powered by Groq's llama3-70b-8192 model
- 🔍 Advanced semantic search using FAISS
- 📊 Efficient document chunking and processing

## Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap 5
- Font Awesome icons

### Backend
- Flask (Python web framework)
- LangChain for RAG pipeline
- HuggingFace Embeddings (BAAI/bge-base-en-v1.5)
- FAISS for vector similarity search
- Groq for LLM inference

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install flask langchain-groq langchain-huggingface langchain-community python-docx docx2txt pypdf faiss-cpu
```

3. Set up your Groq API key:
```bash
export GROQ_API_KEY="your-api-key-here"
```

## Project Structure

```
.
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   └── index.html     # Main interface
├── uploads/           # Document storage (created automatically)
└── README.md          # This file
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a document:
   - Drag and drop a file into the upload area, or
   - Click the "Choose File" button to select a file
   - Supported formats: PDF, DOC, DOCX, TXT

4. Ask questions:
   - Type your question in the input field
   - Press Enter to submit
   - The system will analyze the document and provide relevant answers

## Features in Detail

### Document Processing
- Automatic text extraction from various file formats
- Smart text chunking with 1000-character chunks and 200-character overlap
- Vector embeddings using BAAI/bge-base-en-v1.5
- Efficient similarity search using FAISS

### Question Answering
- Context-aware responses using RAG
- Temperature control (0.7) for balanced outputs
- Top-4 most relevant chunks used for context
- Custom prompt template for accurate answers

### User Interface
- Responsive design that works on all devices
- Dark/Light mode toggle
- Real-time upload progress feedback
- Smooth animations and transitions
- Error handling with user-friendly messages

## Security Features

- Secure filename handling
- File type validation
- Automatic cleanup of old files
- Error handling and validation

## Error Handling

The system includes comprehensive error handling for:
- Invalid file types
- Upload failures
- Processing errors
- Query errors
- Missing documents
- API failures

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Groq for providing the LLM API
- HuggingFace for embeddings
- LangChain for the RAG framework
- Flask team for the web framework

## Support

For support, please open an issue in the repository or contact the maintainers. 