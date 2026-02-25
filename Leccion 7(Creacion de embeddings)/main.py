"""
LECCIÓN 7: Creación de Embeddings y Text Splitters
--------------------------------------------------
En esta lección aprenderemos a transformar texto plano en vectores (embeddings).
Para ello, primero debemos dividir el texto en fragmentos manejables (chunks)
usando un 'Text Splitter', y luego convertirlos en representaciones numéricas.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
# Importamos el splitter avanzado
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv(find_dotenv())

# 2. INICIALIZACIÓN
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 3. CARGA DEL DOCUMENTO
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "ud5_prog.pdf")

print(f"--- 1. CARGANDO DOCUMENTO ---")
if not os.path.exists(file_path):
    print(f"ERROR: No se encuentra el archivo {file_path}")
else:
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    print(f"Documento cargado. Páginas: {len(pages)}")

    # Unimos todo el contenido para demostrar la división manual
    contenido_completo = " ".join([page.page_content for page in pages])
    print(f"Total caracteres: {len(contenido_completo)}\n")

    # 4. DIVISIÓN DE TEXTO (TEXT SPLITTING)
    # El splitter 'RecursiveCharacterTextSplitter' es el recomendado.
    # Intenta mantener párrafos y oraciones juntos antes de cortar.
    print(f"--- 2. DIVIDIENDO TEXTO EN CHUNKS ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Tamaño máximo de cada fragmento (caracteres)
        chunk_overlap=200,    # Solapamiento para no perder el contexto entre cortes
        length_function=len,  # Función para contar el tamaño (len cuenta caracteres)
        is_separator_regex=False, # Indica si los separadores son expresiones regulares
    )

    # NOTA PARA ESTUDIANTES:
    # - length_function=len: Usamos la función nativa de Python para contar caracteres.
    #   Podríamos usar otras para contar tokens (más avanzado).
    # - is_separator_regex=False: Los separadores (., \n, etc.) se tratan como texto 
    #   literal, no como patrones complejos de búsqueda.

    chunks = text_splitter.split_text(contenido_completo)

    print(f"El texto se ha dividido en {len(chunks)} fragmentos.")
    print(f"Ejemplo del primer fragmento:\n{chunks[0][:200]}...\n")

    # 5. GENERACIÓN DE EMBEDDINGS
    # Los embeddings convierten el texto en una lista de números (vectores)
    # que representan el significado semántico.
    print(f"--- 3. GENERANDO EMBEDDINGS ---")
    try:
        # Generamos los embeddings para todos nuestros fragmentos
        # Nota: Esto consume tokens de OpenAI
        embedded_docs = embeddings_model.embed_documents(chunks)
        
        print(f"Se han generado {len(embedded_docs)} vectores.")
        print(f"Dimensión de cada vector: {len(embedded_docs[0])}")
        print(f"Vista previa del primer vector (primeros 5 valores):\n{embedded_docs[0][:5]}")

        # NOTA SOBRE DIMENSIONES PARA ESTUDIANTES:
        # - ¿Por qué 1536?: Es la dimensión por defecto de los modelos de OpenAI (ada-002 y 3-small).
        # - ¿Cómo se cambia?: 
        #   1. Cambiando el modelo: Cada modelo tiene su propia dimensión (ej. 384 en modelos ligeros).
        #   2. Parámetro 'dimensions': Los modelos nuevos de OpenAI (v3) permiten reducir la 
        #      dimensión (ej. de 3072 a 256 o 1024) al inicializar el objeto OpenAIEmbeddings.
        # - ¿Cuál es mejor?:
        #   * Dimensiones GRANDES: Mayor precisión y matices, pero más lentas y costosas de almacenar.
        #   * Dimensiones PEQUEÑAS: Más rápidas y baratas, pero con riesgo de menor precisión.
        
        print("\n¡ÉXITO! Ahora el texto es comprensible para algoritmos de búsqueda matemática.")
    except Exception as e:
        print(f"Error al generar embeddings: {e}")
        print("Asegúrate de tener saldo en tu cuenta de OpenAI y la API Key correcta.")