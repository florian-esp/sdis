# Description

Cette application permet de discuter avec un chatbot utilisant des sources documentaires comme des procédures ou manuels.
Elle utilise les modèles llama pour procéder les documents et répondre à une question.
Le module Qdrant est utilisé pour digérer les documents fournis et les rendre accessible au chatbot

# Installer l'application

L'app requière:
    - Qdrant
    - llama (modèles: llama3, llama3.2-vision, nomic-text-embed)
    - des modules Python (streamlit, langchain, qdrant)

Qdrant est installé dans un conteneur docker dont un fichier d'installation est fourni (docker_install.sh)

# Lancement de l'application

Dans un premier terminal:
    - docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

Dans un second terminal:
    - streamlit run app.py