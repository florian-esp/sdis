import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_classic.chains import RetrievalQA

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "base_connaissances"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="SDIS - Qdrant & Ollama", layout="wide")

@st.cache_resource
def get_models():
    # Embeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    # LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return embeddings, llm

embeddings, llm = get_models()

def process_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
            docs.extend(loader.load())
        finally:
            os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def index_data(splits):
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
    
    QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=False
    )

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["Ajouter des Fichiers", "Discuter (Chat)"])

if page == "Ajouter des Fichiers":
    st.title("Alimenter la Base de fichiers")
    uploaded_files = st.file_uploader("Sélectionner des fichiers", accept_multiple_files=True, type=['pdf', 'txt'])

    if st.button("Ajouter à la base de fichiers"):
        if not uploaded_files:
            st.warning("Veuillez choisir un fichier.")
        else:
            with st.spinner("Lecture et Vectorisation en cours..."):
                try:
                    splits = process_documents(uploaded_files)
                    index_data(splits)
                    st.success(f"Succès ! {len(splits)} segments ajoutés.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    
    if st.button("Réinitialiser la base de fichiers"):
        try:
            client = QdrantClient(url=QDRANT_URL)
            client.delete_collection(COLLECTION_NAME)
            st.warning("Base effacée.")
        except:
            pass

elif page == "Discuter (Chat)":
    st.title("Assistant IA Local")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Votre question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=embeddings,
                    collection_name=COLLECTION_NAME,
                    url=QDRANT_URL
                )
                
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )

                with st.spinner("Recherche..."):
                    response = qa_chain.invoke({"query": prompt})
                
                result_text = response["result"]
                
                st.markdown(result_text)
                
                with st.expander("Sources"):
                    for doc in response["source_documents"]:
                        st.caption(doc.page_content[:200] + "...")

                st.session_state.messages.append({"role": "assistant", "content": result_text})

            except Exception as e:
                st.error("Erreur.")
                st.info(f"Détail : {e}")



