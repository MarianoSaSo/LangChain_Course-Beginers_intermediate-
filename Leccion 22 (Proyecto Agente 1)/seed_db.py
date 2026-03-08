"""
SCRIPT DE APOYO: Carga de Documentos a Pinecone
----------------------------------------------
Este script se encarga de leer el PDF de "Historia de España",
dividirlo en fragmentos y subirlos a nuestro índice de Pinecone.

⚠️ IMPORTANTE: Ejecuta este script UNA SOLA VEZ antes de iniciar el agente
en el archivo 'main.py'. Esto "alimentará" tu base de datos vectorial para que
el agente tenga conocimientos específicos sobre España.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# 1. SETUP
load_dotenv()
embeddings_model = OpenAIEmbeddings()

# 2. CARGA DEL PDF
print("--- 1. CARGANDO Y DIVIDIENDO DOCUMENTO ---")
# Obtenemos la ruta absoluta de la carpeta donde está este script
base_dir = os.path.dirname(os.path.abspath(__file__))
path_pdf = os.path.join(base_dir, "Historia de Espana.pdf")

loader = PyPDFLoader(path_pdf)
pages = loader.load_and_split()

# Dividimos en trozos (chunks) para que la búsqueda sea más precisa
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Unimos páginas y dividimos
contenido = "\n".join([p.page_content for p in pages])
chunks = text_splitter.split_text(contenido)
print(f"Documento dividido en {len(chunks)} fragmentos.")

# 3. CONFIGURACIÓN DE PINECONE
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchain-curso"

# Si el índice no existe (aunque debería estar de la lección 9), lo creamos
if index_name not in pc.list_indexes().names():
    print(f"Creando índice {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 4. SUBIDA DE DATOS (UPSERT)
print("--- 2. SUBIENDO VECTORES A PINECONE ---")
embedded_chunks = embeddings_model.embed_documents(chunks)

vectors_to_upsert = [
    {
        "id": f"historia_espana_{i}",
        "values": embedded_chunks[i],
        "metadata": {"text": chunks[i]}
    }
    for i in range(len(chunks))
]

index.upsert(vectors=vectors_to_upsert)
print("¡Éxito! La base de datos vectorial ha sido actualizada con la Historia de España.")
