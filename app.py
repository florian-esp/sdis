import streamlit as st
import tempfile
import os

# Importations LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient, models

# --- CONFIGURATION ---
QDRANT_URL = "http://localhost:6333" # URL par défaut pour l'install .deb
COLLECTION_NAME = "knowledge_base"   # Nom de votre base de savoir
LLM_MODEL = "llama3"                 # Assurez-vous d'avoir fait 'ollama pull llama3'
EMBED_MODEL = "nomic-embed-text"     # Assurez-vous d'avoir fait 'ollama pull nomic-embed-text'

# --- FONCTIONS UTILITAIRES ---

@st.cache_resource
def get_models():
    """Charge les modèles une seule fois pour la performance."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return embeddings, llm

def process_documents(uploaded_files):
    """Lit, découpe et transforme les fichiers en vecteurs."""
    docs = []
    for file in uploaded_files:
        # Création d'un fichier temporaire pour lecture
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
            os.remove(tmp_path) # Nettoyage

    # Découpage du texte (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def index_data(splits, embeddings):
    """Envoie les données vers Qdrant."""
    client = QdrantClient(url=QDRANT_URL)
    
    # Vérifie si la collection existe, sinon la crée
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )

    # Ajout des documents (sans écraser l'existant, on ajoute à la suite)
    QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=False 
    )

# --- INTERFACE PRINCIPALE ---

st.set_page_config(page_title="RAG Local Qdrant", layout="wide")
embeddings, llm = get_models()

# Menu de Navigation (Sidebar)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["📁 Gestion des Fichiers", "🤖 Chatbot"])

# ---------------- PAGE 1 : GESTION DES FICHIERS ----------------
if page == "📁 Gestion des Fichiers":
    st.title("📁 Alimentation de la Base de Connaissances")
    st.markdown("Ici, vous pouvez ajouter des documents (PDF ou TXT) que l'IA utilisera.")

    uploaded_files = st.file_uploader("Déposez vos fichiers ici", accept_multiple_files=True, type=['pdf', 'txt'])

    if st.button("Ajouter à la base Qdrant"):
        if not uploaded_files:
            st.warning("Veuillez sélectionner des fichiers.")
        else:
            with st.spinner("Traitement et indexation en cours..."):
                try:
                    # 1. Traitement
                    splits = process_documents(uploaded_files)
                    
                    # 2. Indexation
                    index_data(splits, embeddings)
                    
                    st.success(f"Succès ! {len(splits)} nouveaux segments de texte ont été ajoutés à Qdrant.")
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")

    # Optionnel : Bouton pour vider la base
    st.divider()
    if st.button("⚠️ Effacer toute la mémoire (Reset)", type="primary"):
        client = QdrantClient(url=QDRANT_URL)
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            st.warning("La base de données a été effacée.")
        else:
            st.info("La base est déjà vide.")

# ---------------- PAGE 2 : CHATBOT ----------------
elif page == "🤖 Chatbot":
    st.title("🤖 Assistant Documentaire")
    
    # Initialisation de l'historique de chat visuel
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez votre question sur les documents..."):
        # 1. Afficher le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Générer la réponse
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Recherche dans Qdrant...")
            
            try:
                # Connexion à Qdrant
                vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=embeddings,
                    collection_name=COLLECTION_NAME,
                    url=QDRANT_URL
                )
                
                # Configuration du retriever (cherche les 4 morceaux les plus pertinents)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                
                # Chaîne de réponse
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                
                # Exécution
                response = qa_chain.invoke({"query": prompt})
                result_text = response["result"]
                
                # Affichage de la réponse
                message_placeholder.markdown(result_text)
                
                # Affichage des sources (optionnel, dans un menu déroulant)
                with st.expander("Sources utilisées"):
                    for doc in response["source_documents"]:
                        st.caption(f"📄 Contenu : {doc.page_content[:200]}...")
                
                # Sauvegarde dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": result_text})

            except Exception as e:
                message_placeholder.error("Erreur : La base de données semble vide ou inaccessible.")
                st.error(f"Détail technique : {e}")

