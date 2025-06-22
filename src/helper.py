# langchain_community is now used for document loaders
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# langchain_text_splitters is the new home for text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# This function logic remains the same
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",  # 'globs' is now 'glob' in newer versions
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    
    return documents

# This function logic remains the same
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# This function is already correct
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings