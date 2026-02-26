"""
LECCIÓN 9: Almacenamiento en Bases de Datos Vectoriales (Pinecone) + Chat + Memoria
--------------------------------------------------------------------------------
En esta lección aprenderemos a persistir nuestros embeddings en una base de datos
vectorial (Pinecone) y a gestionar el contexto de una conversación (Memoria).

Conceptos clave:
1. RAG (Retrieval Augmented Generation): Recuperar información de documentos.
2. Memoria: Permitir que el LLM recuerde preguntas y respuestas anteriores.
3. Historial de Mensajes: Gestión manual del flujo de conversación.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage

# 1. CONFIGURACIÓN DEL ENTORNO
# Cargamos las variables de entorno para API Keys (OpenAI y Pinecone)
load_dotenv(find_dotenv())

# 1.2 HISTORIAL DE CONVERSACIÓN (MEMORIA)
# --- NOTA TEÓRICA PARA ESTUDIANTES ---
# En versiones antiguas de LangChain se usaba 'langchain.memory'. 
# Hoy en día, ese módulo se considera "Legacy" (antiguo).
# La forma moderna y recomendada es usar 'ChatMessageHistory' porque:
# 1. Da más control al programador sobre qué se guarda y qué no.
# 2. Es compatible con las versiones más recientes de la librería.
# 3. Facilita la transición a herramientas avanzadas como LangGraph.
history = ChatMessageHistory()

# 2. INICIALIZACIÓN DE MODELOS
# El modelo de chat será el encargado de generar respuestas.
chat = ChatOpenAI(model="gpt-4o")
# El modelo de embeddings convertirá nuestros textos en vectores (números).
embeddings_model = OpenAIEmbeddings()

# 3. CARGA Y DIVISIÓN DEL DOCUMENTO
# Buscamos el archivo PDF en la misma carpeta que el script.
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "carta.pdf")

print(f"--- 1. PREPARANDO DOCUMENTO ---")
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

# Unimos el contenido y lo dividimos en fragmentos (chunks) manejables.
contenido_completo = "\n".join([page.page_content for page in pages])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(contenido_completo)

print(f"Documento dividido en {len(chunks)} fragmentos.")

# 4. CONFIGURACIÓN DE PINECONE (Base de Datos Vectorial)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchain-curso"

# Verificamos si el índice existe, si no, lo creamos.
if index_name not in pc.list_indexes().names():
    print(f"Creando nuevo índice: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1536, # Dimensión estándar para embeddings de OpenAI
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 5. SUBIDA DE DATOS (UPSERT)
# Convertimos los fragmentos de texto en vectores y los subimos a Pinecone.
print(f"--- 2. SUBIENDO VECTORES A PINECONE ---")
embedded_chunks = embeddings_model.embed_documents(chunks)

vectors_to_upsert = [
    {
        "id": f"chunk_{i}",
        "values": embedded_chunks[i],
        "metadata": {"text": chunks[i]} # Guardamos el texto original para recuperarlo después
    }
    for i in range(len(chunks))
]

index.upsert(vectors=vectors_to_upsert)
print(f"Vectores subidos con éxito.")

# 6. SISTEMA DE RESPUESTA (RAG) + MEMORIA
print(f"\n--- 3. CONSULTA CON MEMORIA (RAG) ---")

def responder_pregunta(pregunta_usuario):
    """
    Esta función hace tres cosas:
    1. Busca información relevante en Pinecone (Contexto).
    2. Recupera la conversación anterior (Memoria).
    3. Genera la respuesta usando el LLM.
    """
    
    # a) BÚSQUEDA SEMÁNTICA: Buscamos información en el PDF
    query_vector = embeddings_model.embed_query(pregunta_usuario)
    busqueda = index.query(vector=query_vector, top_k=2, include_metadata=True)
    contexto = "\n\n".join([match["metadata"]["text"] for match in busqueda["matches"]])
    
    # b) PROMPT DE SISTEMA: Instrucciones y Contexto
    system_prompt = (
        "Eres un asistente experto. Utiliza el contexto para responder. "
        "Si no sabes la respuesta, dilo.\n\n"
        f"CONTEXTO EXTRAÍDO DEL DOCUMENTO:\n{contexto}"
    )
    
    # c) CONSTRUCCIÓN DE MENSAJES (Estructura de memoria)
    # Combinamos: Instrucciones + Historial Pasado + Pregunta del Presente
    mensajes = [SystemMessage(content=system_prompt)]
    mensajes.extend(history.messages) # Lo que ya hablamos
    mensajes.append(HumanMessage(content=pregunta_usuario)) # Lo que preguntas ahora
    
    # d) GENERAR RESPUESTA
    respuesta = chat.invoke(mensajes)
    
    # e) ACTUALIZAR MEMORIA: Guardamos lo ocurrido para la próxima pregunta
    history.add_user_message(pregunta_usuario)
    history.add_ai_message(respuesta.content)
    
    return respuesta.content

# --- 7. BUCLE INTERACTIVO PARA ESTUDIANTES ---
# Permite probar que el chat "recuerda" preguntando cosas sobre mensajes anteriores.
if __name__ == "__main__":
    print("\nBienvenido al Chat con Memoria de la Lección 9.")
    print("Prueba preguntando: '¿Quién escribe la carta?' y después '¿Cómo me llamo?'")
    
    while True:
        pregunta = input("\nEscribe tu pregunta (o 'salir' para terminar): ")
        if pregunta.lower() == "salir":
            print("Programa terminado.")
            break
        
        respuesta = responder_pregunta(pregunta)
        print(f"\nRespuesta: {respuesta}")
        print("-" * 50)
