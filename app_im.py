import streamlit as st
import tempfile
import os
import base64
import fitz  
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


from langchain_classic.chains import RetrievalQA 


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "base_connaissances"
LLM_MODEL = "llama3" 
VISION_MODEL = "llava" 
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="SDIS - Qdrant & Ollama", layout="wide")

@st.cache_resource
def get_models():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    vision_llm = ChatOllama(model=VISION_MODEL, temperature=0)
    return embeddings, llm, vision_llm

embeddings, llm, vision_llm = get_models()


def describe_image(image_path):
    """
    Envoie une image au modèle Vision pour obtenir une description textuelle.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        
        prompt = "Décris cette image en détail pour un contexte professionnel. Identifie les objets, les situations de danger, les textes visibles et le matériel technique."
        
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
        )
        response = vision_llm.invoke([msg])
        return response.content
    except Exception as e:
        return f"[Erreur analyse image : {e}]"


def process_pdf_with_images(file_path, filename):
    doc = fitz.open(file_path)
    documents = []

    for page_num, page in enumerate(doc):
        text_content = page.get_text()
        
        
        image_list = page.get_images(full=True)
        image_descriptions = []
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            
            if len(image_bytes) < 5000:
                continue
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{base_image['ext']}") as tmp_img:
                tmp_img.write(image_bytes)
                tmp_img_path = tmp_img.name
            
            try:
                desc = describe_image(tmp_img_path)
                image_descriptions.append(f"\n[DESCRIPTION IMAGE PAGE {page_num+1}] : {desc}\n")
            finally:
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)

        full_page_content = text_content + "\n".join(image_descriptions)
        
        documents.append(Document(
            page_content=full_page_content,
            metadata={"source": filename, "page": page_num + 1}
        ))
        
    return documents


def process_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        file_ext = file.name.split('.')[-1].lower()
        suffix = f".{file_ext}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if file_ext == "pdf":
                st.info(f"Traitement complet (Texte+Images) de {file.name}...")
                docs.extend(process_pdf_with_images(tmp_path, file.name))
            
            elif file_ext in ["png", "jpg", "jpeg"]:
                st.info(f"Analyse vision de {file.name}...")
                desc = describe_image(tmp_path)
                docs.append(Document(
                    page_content=f"[IMAGE SOURCE: {file.name}]\nDescription: {desc}",
                    metadata={"source": file.name, "type": "image"}
                ))
            
            elif file_ext == "docx":
                loader = Docx2txtLoader(tmp_path)
                docs.extend(loader.load())
            
            else:
                loader = TextLoader(tmp_path)
                docs.extend(loader.load())

        except Exception as e:
            st.error(f"Erreur sur {file.name} : {e}")
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
        splits, embeddings, url=QDRANT_URL, collection_name=COLLECTION_NAME, force_recreate=False
    )


st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["Ajouter des Fichiers", "Discuter (Chat)"])

if page == "Ajouter des Fichiers":
    st.title("Alimenter la Base de fichiers")
    uploaded_files = st.file_uploader(
        "Sélectionner des fichiers", 
        accept_multiple_files=True, 
        type=['pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg']
    )
    
    if st.button("Ajouter à la base de fichiers"):
        if not uploaded_files:
            st.warning("Veuillez choisir un fichier.")
        else:
            with st.spinner("Lecture et Vectorisation (l'analyse d'images peut prendre du temps)..."):
                try:
                    splits = process_documents(uploaded_files)
                    if splits:
                        index_data(splits)
                        st.success(f"Succès ! {len(splits)} segments ajoutés.")
                    else:
                        st.warning("Aucun contenu extrait.")
                except Exception as e:
                    st.error(f"Erreur globale : {e}")

    if st.button("Réinitialiser la base de fichiers"):
        try:
            client = QdrantClient(url=QDRANT_URL)
            client.delete_collection(COLLECTION_NAME)
            st.warning("Base effacée.")
        except Exception as e:
            st.error(f"Erreur lors de la suppression : {e}")

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
                    embedding=embeddings, collection_name=COLLECTION_NAME, url=QDRANT_URL
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                
                custom_template = """Tu es un assistant francophone utile et précis. 
                Utilise les éléments de contexte suivants pour répondre à la question à la fin.
                Note : Le contexte contient des descriptions d'images analysées par IA.
                Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
                
                Contexte: {context}
                Question : {question}
                Réponse utile:"""
                
                QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_template)
                
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt":QA_CHAIN_PROMPT}
                )
                
                with st.spinner("Recherche..."):
                    response = qa_chain.invoke({"query": prompt})
                    result_text = response["result"]
                    st.markdown(result_text)
                    
                    with st.expander("Sources"):
                        for doc in response["source_documents"]:
                            source_name = doc.metadata.get('source', 'Inconnue')
                            if doc.metadata.get("type") == "image":
                                st.caption(f"**Source Image:** {os.path.basename(source_name)}")
                            else:
                                st.caption(f"**Source:** {os.path.basename(source_name)}")
                            st.caption(doc.page_content[:200] + "...")
                            
                    st.session_state.messages.append({"role": "assistant", "content": result_text})
            except Exception as e:
                st.error("Erreur lors de la génération de la réponse.")
                st.info(f"Détail : {e}")
