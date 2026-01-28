import streamlit as st
import tempfile
import os

# --- Imports Standards ---
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

# --- Imports pour l'OCR MANUEL (Plus stable) ---
from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_path

# --- CONFIGURATION ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "base_connaissances"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="RAG - OCR Manuel", layout="wide")

# --- INITIALISATION ---
@st.cache_resource
def get_models():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return embeddings, llm

embeddings, llm = get_models()

# --- FONCTIONS OCR MANUELLES ---

def ocr_image(image_path_or_bytes):
    """Extrait le texte d'une image."""
    engine = RapidOCR()
    # RapidOCR retourne une liste de résultats
    result, _ = engine(image_path_or_bytes)
    if not result:
        return ""
    # On rassemble tout le texte trouvé
    return "\n".join([line[1] for line in result])

def process_pdf_with_ocr(pdf_path):
    """Convertit PDF -> Images -> Texte via OCR."""
    documents = []
    # 1. Convertir les pages du PDF en images
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        raise Exception(f"Erreur poppler/pdf2image : {e}")

    # 2. Analyser chaque page
    for i, img in enumerate(images):
        # On sauvegarde l'image temporairement pour RapidOCR ou on lui passe des bytes
        # RapidOCR accepte les tableaux numpy ou chemins, on va passer par bytes pour faire simple
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        text = ocr_image(img_bytes)
        
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={"source": pdf_path, "page": i+1, "type": "pdf_ocr"}
            )
            documents.append(doc)
            
    return documents

# --- TRAITEMENT GÉNÉRAL ---
def process_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        file_ext = file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            # 1. PDF (Méthode Manuelle Robuste)
            if file.name.lower().endswith(".pdf"):
                with st.spinner(f"OCR page par page sur {file.name}..."):
                    pdf_docs = process_pdf_with_ocr(tmp_path)
                    docs.extend(pdf_docs)

            # 2. WORD
            elif file.name.lower().endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
                docs.extend(loader.load())
            
            # 3. IMAGES
            elif file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                text_content = ocr_image(tmp_path)
                if text_content.strip():
                    docs.append(Document(page_content=text_content, metadata={"source": file.name}))
            
            # 4. TEXTE
            else:
                loader = TextLoader(tmp_path)
                docs.extend(loader.load())

        except Exception as e:
            st.error(f"Erreur sur {file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
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

# --- INTERFACE ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Menu", ["Ajout Documents", "Chatbot"])

if page == "Ajout Documents":
    st.title("Ajouter des documents (OCR activé)")
    uploaded_files = st.file_uploader("Fichiers", accept_multiple_files=True)

    if st.button("Indexer"):
        if uploaded_files:
            try:
                splits = process_documents(uploaded_files)
                if splits:
                    index_data(splits)
                    st.success(f"Fait ! {len(splits)} morceaux indexés.")
                else:
                    st.warning("Rien à indexer.")
            except Exception as e:
                st.error(f"Erreur : {e}")

    if st.button("Vider la base"):
        try:
            QdrantClient(url=QDRANT_URL).delete_collection(COLLECTION_NAME)
            st.warning("Base vidée.")
        except: pass

elif page == "Chatbot":
    st.title("Chatbot")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Question ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                vector_store = QdrantVectorStore.from_existing_collection(embeddings, COLLECTION_NAME, url=QDRANT_URL)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
                
                with st.spinner("..."):
                    res = qa.invoke({"query": prompt})
                
                st.markdown(res["result"])
                with st.expander("Sources"):
                    for doc in res["source_documents"]:
                        st.caption(f"{doc.metadata.get('source')} (Page {doc.metadata.get('page', '?')})")
                
                st.session_state.messages.append({"role": "assistant", "content": res["result"]})
            except Exception as e:
                st.error(e)


