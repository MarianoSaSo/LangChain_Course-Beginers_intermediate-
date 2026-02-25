"""
LECCIÓN 8: Almacenamiento en Bases de Datos Vectoriales (Pinecone)
------------------------------------------------------------------
En esta lección aprenderemos a persistir nuestros embeddings en una base de datos
vectorial (Vector Store). Esto es fundamental para construir sistemas de RAG 
(Retrieval Augmented Generation), permitiendo que la IA consulte documentos externos.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv(find_dotenv())

# 2. INICIALIZACIÓN DE MODELOS
# Usamos OpenAI para generar los embeddings y para el chat
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 3. CARGA Y DIVISIÓN DEL DOCUMENTO
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "carta.pdf")

print(f"--- 1. PREPARANDO DOCUMENTO ---")
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

# Unimos y dividimos en chunks manejables
contenido_completo = "\n".join([page.page_content for page in pages])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(contenido_completo)

print(f"Documento dividido en {len(chunks)} fragmentos.")

# 4. CONFIGURACIÓN DE PINECONE
# Pinecone es una base de datos optimizada para buscar vectores por similitud.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchain-curso"

# Creamos el índice si no existe
# NOTA: Los embeddings de OpenAI (text-embedding-ada-002 / 3-small) tienen 1536 dimensiones.
if index_name not in pc.list_indexes().names():
    print(f"Creando nuevo índice: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine", # Métrica de similitud clásica
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 5. SUBIDA DE DATOS (UPSERT)
# Convertimos el texto en embeddings y los subimos a Pinecone junto con su metadata
print(f"--- 2. SUBIENDO VECTORES A PINECONE ---")

# Generamos los embeddings
embedded_chunks = embeddings_model.embed_documents(chunks)

# Preparamos el formato que requiere Pinecone: (id, vector, metadata)
vectors_to_upsert = [
    {
        "id": f"chunk_{i}",
        "values": embedded_chunks[i],
        "metadata": {"text": chunks[i]} # Guardamos el texto original para recuperarlo después
    }
    for i in range(len(chunks))
]

index.upsert(vectors=vectors_to_upsert)
print(f"Vectores subidos con éxito al índice '{index_name}'.")

# 6. SISTEMA DE RECUPERACIÓN (RAG)
print(f"\n--- 3. CONSULTA AL SISTEMA (RAG) ---")

def responder_con_contexto(pregunta_usuario):
    # a) Convertimos la pregunta del usuario en un vector
    query_vector = embeddings_model.embed_query(pregunta_usuario)
    
    # b) Buscamos en Pinecone los fragmentos más parecidos
    busqueda = index.query(vector=query_vector, top_k=2, include_metadata=True)
    
    # c) Extraemos el texto de los resultados
    contexto = "\n\n".join([match["metadata"]["text"] for match in busqueda["matches"]])
    
    # d) Generamos la respuesta final usando el LLM y el contexto recuperado
    print(f"Contexto recuperado de Pinecone:\n{contexto[:200]}...\n")
    
    system_prompt = (
        "Eres un asistente experto. Utiliza el siguiente contexto para responder "
        "la pregunta. Si no lo sabes, di que no tienes esa información.\n\n"
        f"CONTEXTO:\n{contexto}"
    )
    
    from langchain_core.messages import SystemMessage, HumanMessage
    mensajes = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=pregunta_usuario)
    ]
    
    respuesta = chat.invoke(mensajes)
    return respuesta.content

# Ejemplo de uso práctico:
pregunta = "¿De qué ciudad es la empresa mencionada en la carta?"
print(f"PREGUNTA: {pregunta}")
resultado = responder_con_contexto(pregunta)
print(f"RESPUESTA FINAL:\n{resultado}")