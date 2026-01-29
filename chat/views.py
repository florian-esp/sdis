from django.shortcuts import render
from .app import get_rag_chain, process_and_index_django

def chat_view(request):
    reponse = ""
    if request.method == "POST":
        if 'question' in request.POST:
            # Partie Chat
            question = request.POST.get("question")
            try:
                chain = get_rag_chain()
               
                res = chain.invoke({"input": question})
                
                reponse = res["answer"] 
            except Exception as e:
                reponse = f"Erreur lors de la génération : {e}"
            
        elif 'file' in request.FILES:
            # Partie Upload
            try:
                uploaded_file = request.FILES['file']
                nb_segments = process_and_index_django(uploaded_file)
                reponse = f"Fichier indexé avec succès ({nb_segments} segments)."
            except Exception as e:
                reponse = f"Erreur lors de l'indexation : {e}"

    return render(request, "chat/chat.html", {"reponse": reponse})