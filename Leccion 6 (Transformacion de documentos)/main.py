"""
LECCIÓN 6: Transformación de Documentos
-----------------------------------------
En esta lección profundizaremos en cómo manipular el contenido extraído de los
documentos. Aprenderemos a seleccionar fragmentos específicos y a preparar
la información para que el modelo pueda procesarla de forma eficiente.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_community.document_loaders import PyPDFLoader

# 1. CONFIGURACIÓN DEL ENTORNO
# Usamos find_dotenv para localizar el archivo .env automáticamente
load_dotenv(find_dotenv())

# 2. INICIALIZACIÓN DEL MODELO
# Configuramos el modelo gpt-4o. Recuerda que este modelo tiene una ventana 
# de contexto más amplia, pero sigue teniendo límites.
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# 3. CARGA DEL DOCUMENTO
# Obtenemos la ruta absoluta al archivo PDF dentro de la carpeta de la lección
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ud5_prog.pdf")

print(f"--- CARGANDO DOCUMENTO ---")
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

print(f"Documento cargado correctamente.")
print(f"Total de páginas detectadas: {len(pages)}\n")

# 4. CONFIGURACIÓN DEL PROMPT
# Definimos una plantilla para pedir resúmenes de forma estructurada
human_template = "Necesito que hagas un resumen conciso del siguiente contenido extraído de un PDF: {contenido}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

# 5. TRANSFORMACIÓN Y PROCESAMIENTO: Resumen de una página específica
# Aquí seleccionamos una página concreta (la página 7 en este caso)
print("--- PROCESANDO PÁGINA 7 ---")
contenido_pagina_7 = pages[7].page_content
solicitud_pag = chat_prompt.format_messages(contenido=contenido_pagina_7)

resultado_pag = chat.invoke(solicitud_pag)
print(f"Resumen de la página 7:\n{resultado_pag.content}\n")

# 6. TRANSFORMACIÓN Y PROCESAMIENTO: Intento de resumen de todo el documento
# En esta etapa, unificamos todas las páginas. 
# NOTA: Si el documento es muy largo, esto podría fallar dependiendo del modelo.
print("--- PROCESANDO TODO EL DOCUMENTO ---")
try:
    contenido_completo = " ".join([page.page_content for page in pages])
    solicitud_completa = chat_prompt.format_messages(contenido=contenido_completo)
    
    resultado_total = chat.invoke(solicitud_completa)
    print(f"Resumen general del documento:\n{resultado_total.content}")
except Exception as e:
    print(f"Error al procesar el documento completo: {e}")
    print("Recordatorio: Si el texto es demasiado grande, necesitaremos 'Text Splitters' (Lección siguiente).")
