from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'txt', 'docx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize LLM and embeddings
llm = ChatGroq(
    model_name="llama3-70b-8192",
    groq_api_key="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",
    temperature=0.7
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create a custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Please provide a detailed and accurate answer based on the context above."""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_document(file_path):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'txt':
        loader = TextLoader(file_path)
    elif file_extension == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension in ['doc', 'docx']:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Clear the uploads directory
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the document and create vector store
            texts = load_document(file_path)
            global vector_store
            vector_store = FAISS.from_documents(texts, embeddings)
            
            # Create the QA chain
            global qa_chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            return jsonify({'message': 'File uploaded and processed successfully'}), 200
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        if 'qa_chain' not in globals():
            return jsonify({'error': 'Please upload a document first'}), 400
        
        # Get response from the QA chain
        response = qa_chain.run(query_text)
        return jsonify({'response': response}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 