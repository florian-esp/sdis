from django.shortcuts import render
from django.http import JsonResponse
from .rag_utils import get_rag_chain, process_and_index # importez vos fonctions
import json

def chat_view(request):
    # Initialisation de l'historique en session
    if 'messages' not in request.session:
        request.session['messages'] = []

    if request.method == "POST":
        data = json.loads(request.body)
        user_query = data.get('message')
        
        # Appel à LangChain
        chain = get_rag_chain()
        response = chain.invoke({"query": user_query})
        
        answer = response["result"]
        
        # Sauvegarde en session
        request.session['messages'].append({"role": "user", "content": user_query})
        request.session['messages'].append({"role": "assistant", "content": answer})
        request.session.modified = True
        
        return JsonResponse({"status": "success", "answer": answer})

    return render(request, 'chat.html', {"messages": request.session['messages']})