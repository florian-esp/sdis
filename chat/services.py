from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_community.graphs.graph_document import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "base_connaissances"

def get_rag_chain():  # <-- Vérifie bien ce nom !
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3", temperature=0)
    
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    template = "Tu es un assistant francophone... Contexte: {context} Question : {question} Réponse:"
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )