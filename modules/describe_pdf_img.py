import fitz
from .describe_image import describe_image
import os 
from langchain_core.documents import Document

def describe_pdf_img(file_path, filename, tempfile):
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