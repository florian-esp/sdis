import streamlit as st
import tempfile
import os

# --- Imports Standards ---
from langchain_community.document_loaders import TextLoader, Docx2txtLoader # <--- Ajout de Docx2txtLoader
from langchain_community.document_loaders import RapidOCRPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain.chains import RetrievalQA

# --- Imports Images ---
from langchain.docstore.document import Document
from rapidocr_onnxruntime import RapidOCR

# --- CONFIGURATION ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "base_connaissances"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="RAG Multi-Format", layout="wide")

# --- INITIALISATION ---
@st.cache_resource
def get_models():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return embeddings, llm

embeddings, llm = get_models()

# --- FONCTION OCR (Images seules) ---
def extract_text_from_image(image_path):
    engine = RapidOCR()
    result, _ = engine(image_path)
    if not result:
        return ""
    return "\n".join([line[1] for line in result])

# --- TRAITEMENT DES DOCUMENTS ---
def process_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        file_ext = file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            # 1. PDF (Texte + OCR)
            if file.name.lower().endswith(".pdf"):
                with st.spinner(f"Analyse PDF (OCR) : {file.name}..."):
                    loader = RapidOCRPDFLoader(tmp_path)
                    docs.extend(loader.load())

            # 2. WORD (.docx)
            elif file.name.lower().endswith(".docx"):
                # Pas besoin de spinner, c'est très rapide
                loader = Docx2txtLoader(tmp_path)
                docs.extend(loader.load())
            
            # 3. IMAGES (JPG/PNG)
            elif file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                st.info(f"Analyse image : {file.name}")
                text_content = extract_text_from_image(tmp_path)
                if text_content.strip():
                    new_doc = Document(
                        page_content=text_content,
                        metadata={"source": file.name, "type": "image"}
                    )
                    docs.append(new_doc)
            
            # 4. TEXTE (.txt)
            else:
                loader = TextLoader(tmp_path)
                docs.extend(loader.load())

        except Exception as e:
            st.error(f"Erreur sur {file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Découpage
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

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["Ajouter des Fichiers", "Discuter (Chat)"])

# --- PAGE UPLOAD ---
if page == "Ajouter des Fichiers":
    st.title("Alimenter la Base (PDF, DOCX, IMG, TXT)")
    
    # Ajout de 'docx' dans la liste des types autorisés
    uploaded_files = st.file_uploader(
        "Sélectionner des fichiers", 
        accept_multiple_files=True, 
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'docx']
    )

    if st.button("Lancer l'indexation"):
        if not uploaded_files:
            st.warning("Veuillez choisir un fichier.")
        else:
            try:
                splits = process_documents(uploaded_files)
                if splits:
                    index_data(splits)
                    st.success(f"Terminé ! {len(splits)} segments ajoutés.")
                else:
                    st.warning("Aucun contenu exploitable trouvé.")
            except Exception as e:
                st.error(f"Erreur globale : {e}")

    if st.button("Réinitialiser la base"):
        try:
            client = QdrantClient(url=QDRANT_URL)
            client.delete_collection(COLLECTION_NAME)
            st.warning("Base effacée.")
        except:
            pass

# --- PAGE CHAT ---
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
                        meta = doc.metadata.get('source', 'doc')
                        st.caption(f"Source ({meta}): {doc.page_content[:200]}...")

                st.session_state.messages.append({"role": "assistant", "content": result_text})

            except Exception as e:
                st.error("Erreur.")
                st.info(f"Détail : {e}")
