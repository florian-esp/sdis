from langchain_core.messages import HumanMessage
import base64

def describe_image(image_path, vision_llm):
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        
        prompt = """"
            Analyse cette image technique pour un usage professionnel (Sapeurs-Pompiers / SDIS).
            Ta tâche comporte 2 étapes:
                1. TRANSCRIPTION: Lis et transcris tout le texte visible dans l'image, mot pour mot. S'il y a des données chiffrées ou des tableaux, recopie-les.
                2. DESCRIPTION: Décris ce que représente l'image (schéma tactique, équipement, situation de danger).
                Format de réponse attendu: **TEXTE DETECTE :** [Insérer le texte lu ici]
                **ANALYSE VISUELLE :** [Insérer la description ici]     
        """
        
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