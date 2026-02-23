"""
LECCION 4: Cargadores de Datos (Document Loaders)
--------------------------------------------------
En esta lección aprenderemos a importar información externa (PDFs) a nuestro
flujo de LangChain y entenderemos el concepto de 'Límite de Contexto'.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
# Importamos el cargador específico para archivos PDF
from langchain_community.document_loaders import PyPDFLoader

# 1. CONFIGURACIÓN DEL ENTORNO
# Cargamos la API Key desde el archivo .env en la raíz
load_dotenv(find_dotenv())

# 2. INICIALIZACIÓN DEL MODELO
# Usamos GPT-4o para procesar el contenido extraído
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# 3. CARGA DEL DOCUMENTO PDF
# Para que el script funcione en cualquier PC, calculamos la ruta relativa al archivo
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ud5_prog.pdf")

# Inicializamos el cargador con la ruta del PDF
loader = PyPDFLoader(file_path)

# 4. PROCESAMIENTO DEL DOCUMENTO
# 'load_and_split' lee el archivo y crea una lista donde cada elemento es una página
pages = loader.load_and_split()

print(f"--- DOCUMENTO CARGADO ---")
print(f"Número total de páginas: {len(pages)}")
print(f"Contenido inicial de la página 1:\n{pages[0].page_content[:200]}...\n")

# 5. ESTRATEGIA DE PROMPT (Plantilla para resúmenes)
human_template = "Necesito que hagas un resumen del contenido del siguiente fragmento de un PDF: {contenido}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

# 6. EJEMPLO DE ÉXITO: Resumir una página específica
# Los LLM tienen una 'Ventana de Contexto' (capacidad de memoria). 
# Una sola página cabe perfectamente.
print("--- EJEMPLO 1: Resumen de la página 10 ---")
pagina_10 = pages[10].page_content
solicitud_pag_10 = chat_prompt.format_messages(contenido=pagina_10)

resultado_corto = chat.invoke(solicitud_pag_10)
print(f"Resumen página 10: {resultado_corto.content}\n")

# 7. EJEMPLO DE ERROR: El límite de contexto
# Intentar enviar las 72 páginas de golpe provocará un error 'context_length_exceeded'.
# Esto sucede porque el texto es demasiado largo para que el modelo lo procese de una vez.
print("--- EJEMPLO 2: Intento de procesar todo el PDF (Explicación del límite) ---")
try:
    # Unificamos todo el contenido en un solo string gigante
    contenido_completo = "\n\n".join([page.page_content for page in pages])
    solicitud_total = chat_prompt.format_messages(contenido=contenido_completo)
    
    print("Enviando todo el PDF al modelo... (esto fallará)")
    result = chat.invoke(solicitud_total)
    print(result.content)
except Exception as e:
    print(f"ERROR ESPERADO: {e}")
    print("\nNOTA PARA ESTUDIANTES: Este error ocurre porque no podemos enviar libros enteros al prompt.")
    print("Para solucionar esto, en futuras lecciones usaremos 'Vector Stores' y 'RAG'.")