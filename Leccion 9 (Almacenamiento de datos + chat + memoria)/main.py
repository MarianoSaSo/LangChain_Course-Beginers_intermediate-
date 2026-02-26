import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv(find_dotenv())

# 1.2 Crear historial de conversación manual
# En esta versión usamos ChatMessageHistory porque tu entorno no tiene el módulo 'memory'
history = ChatMessageHistory()

# 2. INICIALIZACIÓN DE MODELOS
chat = ChatOpenAI(model="gpt-4o")
embeddings_model = OpenAIEmbeddings()

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
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchain-curso"

# Creamos el índice si no existe
if index_name not in pc.list_indexes().names():
    print(f"Creando nuevo índice: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 5. SUBIDA DE DATOS (UPSERT)
print(f"--- 2. SUBIENDO VECTORES A PINECONE ---")
embedded_chunks = embeddings_model.embed_documents(chunks)

vectors_to_upsert = [
    {
        "id": f"chunk_{i}",
        "values": embedded_chunks[i],
        "metadata": {"text": chunks[i]}
    }
    for i in range(len(chunks))
]

index.upsert(vectors=vectors_to_upsert)
print(f"Vectores subidos con éxito.")

# 6. SISTEMA DE RECUPERACIÓN (RAG) + MEMORIA
print(f"\n--- 3. CONSULTA CON MEMORIA (RAG) ---")

def responder_pregunta(pregunta_usuario):
    # a) Búsqueda en Pinecone
    query_vector = embeddings_model.embed_query(pregunta_usuario)
    busqueda = index.query(vector=query_vector, top_k=2, include_metadata=True)
    contexto = "\n\n".join([match["metadata"]["text"] for match in busqueda["matches"]])
    
    # b) Preparar el Prompt con Historial
    system_prompt = (
        "Eres un asistente experto. Utiliza el contexto para responder. "
        "Si no sabes la respuesta, dilo.\n\n"
        f"CONTEXTO:\n{contexto}"
    )
    
    # Construimos la lista de mensajes: Sistema + Historial + Pregunta Actual
    mensajes = [SystemMessage(content=system_prompt)]
    mensajes.extend(history.messages) # Añadimos mensajes anteriores
    mensajes.append(HumanMessage(content=pregunta_usuario)) # Añadimos pregunta actual
    
    # c) Generar respuesta
    respuesta = chat.invoke(mensajes)
    
    # d) Guardar en el historial manualmente
    history.add_user_message(pregunta_usuario)
    history.add_ai_message(respuesta.content)
    
    return respuesta.content

# --- 7. EJEMPLO DE USO INTERACTIVO ---
if __name__ == "__main__":
    while True:
        pregunta = input("\nEscribe tu pregunta (o 'salir' para terminar): ")
        if pregunta.lower() == "salir":
            print("Programa terminado.")
            break
        
        respuesta = responder_pregunta(pregunta)
        print("Pregunta:", pregunta)
        print("Respuesta:", respuesta)
        print("-" * 40)  # Separador para claridad
