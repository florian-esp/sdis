import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient, models
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "base_connaissances"
def get_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3", temperature=0)
    
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # On définit le template directement
    prompt = PromptTemplate.from_template(
        "Contexte: {context} \nQuestion: {input} \nRéponse:"
    )
    
    # On crée la chaîne manuellement sans passer par le module 'chains' global
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
def process_and_index_django(django_file):
    """Version adaptée pour les fichiers envoyés via un formulaire Django"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 1. Sauvegarde temporaire du fichier
    ext = os.path.splitext(django_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        for chunk in django_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # 2. Chargement
        if ext == ".pdf": loader = PyPDFLoader(tmp_path)
        elif ext == ".docx": loader = Docx2txtLoader(tmp_path)
        else: loader = TextLoader(tmp_path)
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Indexation
        QdrantVectorStore.from_documents(
            splits, embeddings, url=QDRANT_URL, 
            collection_name=COLLECTION_NAME, force_recreate=False
        )
        return len(splits)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)