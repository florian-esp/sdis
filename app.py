import streamlit as st

import tempfile

import os

# --- Imports Modernes LangChain (évite les erreurs d'import) ---

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient, models

# Imports pour la chaîne de réponse (Nouvelle méthode)

from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---

QDRANT_URL = "http://localhost:6333" # Port par défaut de Qdrant installé via .deb

COLLECTION_NAME = "base_connaissances"

LLM_MODEL = "llama3"                 # Modèle de Chat

EMBED_MODEL = "nomic-embed-text"     # Modèle d'Embedding

# --- INITIALISATION ---

st.set_page_config(page_title="RAG Local - Qdrant & Ollama", layout="wide")

@st.cache_resource

def get_models():

    """Charge les modèles (mise en cache pour la rapidité)."""

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    return embeddings, llm

embeddings, llm = get_models()

# --- FONCTIONS UTILITAIRES ---

def process_documents(uploaded_files):

    """Lit et découpe les fichiers."""

    docs = []

    for file in uploaded_files:

        # Fichier temporaire nécessaire pour les Loaders LangChain

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

    # Découpage (Chunks)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    return text_splitter.split_documents(docs)

def index_data(splits):

    """Indexe les données dans Qdrant."""

    client = QdrantClient(url=QDRANT_URL)

    

    # Création de la collection si elle n'existe pas

    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),

        )

    # Ajout des vecteurs

    QdrantVectorStore.from_documents(

        splits,

        embeddings,

        url=QDRANT_URL,

        collection_name=COLLECTION_NAME,

        force_recreate=False # False = on ajoute, on n'écrase pas

    )

# --- NAVIGATION ---

st.sidebar.title("Navigation")

page = st.sidebar.radio("Aller vers :", ["📁 Ajouter des Fichiers", "🤖 Discuter (Chat)"])

# ==========================================

# PAGE 1 : GESTION DES FICHIERS

# ==========================================

if page == "📁 Ajouter des Fichiers":

    st.title("📁 Alimentation de la Base")

    st.markdown("Ajoutez ici vos documents PDF ou TXT.")

    uploaded_files = st.file_uploader("Sélectionner des fichiers", accept_multiple_files=True, type=['pdf', 'txt'])

    if st.button("Indexer dans Qdrant"):

        if not uploaded_files:

            st.warning("Veuillez choisir un fichier.")

        else:

            with st.spinner("Lecture et Vectorisation en cours..."):

                try:

                    # 1. Traitement

                    splits = process_documents(uploaded_files)

                    # 2. Indexation

                    index_data(splits)

                    st.success(f"Succès ! {len(splits)} segments ajoutés à la base de données.")

                except Exception as e:

                    st.error(f"Erreur : {e}")

    st.divider()

    # Bouton pour vider la base (utile pour les tests)

    if st.button("⚠️ Vider la mémoire (Reset)", type="primary"):

        try:

            client = QdrantClient(url=QDRANT_URL)

            client.delete_collection(COLLECTION_NAME)

            st.warning("La base de connaissances a été supprimée.")

        except Exception as e:

            st.error(f"Erreur lors du reset : {e}")

# ==========================================

# PAGE 2 : CHATBOT RAG

# ==========================================

elif page == "🤖 Discuter (Chat)":

    st.title("🤖 Assistant IA Local")

    # Gestion de l'historique visuel

    if "messages" not in st.session_state:

        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Zone de saisie

    if prompt := st.chat_input("Posez une question sur vos documents..."):

        # 1. Affiche la question utilisateur

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):

            st.markdown(prompt)

        # 2. Génération de la réponse

        with st.chat_message("assistant"):

            try:

                # Connexion à la base existante

                vector_store = QdrantVectorStore.from_existing_collection(

                    embedding=embeddings,

                    collection_name=COLLECTION_NAME,

                    url=QDRANT_URL

                )

                

                # Le Retriever cherche les infos

                retriever = vector_store.as_retriever(search_kwargs={"k": 4})

                # --- CONSTRUCTION DE LA CHAINE (Nouvelle Syntaxe) ---

                

                # Le Prompt système qui force l'IA à utiliser le contexte

                prompt_template = ChatPromptTemplate.from_template("""

                Tu es un assistant précis. Utilise les éléments de contexte suivants pour répondre à la question.

                Si tu ne connais pas la réponse d'après le contexte, dis simplement que tu ne sais pas.

                

                <contexte>

                {context}

                </contexte>

                Question: {input}

                """)

                # Chaîne 1 : Créer la réponse à partir des docs

                document_chain = create_stuff_documents_chain(llm, prompt_template)

                

                # Chaîne 2 : Récupérer les docs + Créer la réponse

                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Exécution

                with st.spinner("Recherche dans les documents..."):

                    response = retrieval_chain.invoke({"input": prompt})

                

                answer = response['answer']

                

                # Affichage réponse

                st.markdown(answer)

                

                # Affichage des sources (Expandable)

                with st.expander("Voir les sources utilisées"):

                    for i, doc in enumerate(response["context"]):

                        st.caption(f"Source {i+1} : {doc.page_content[:200]}...")

                # Sauvegarde historique

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:

                st.error("Erreur : Impossible de répondre. Avez-vous indexé des documents ?")

                st.info(f"Détail technique : {e}")

