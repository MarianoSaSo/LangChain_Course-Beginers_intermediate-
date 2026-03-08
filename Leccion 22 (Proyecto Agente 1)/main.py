"""
PROYECTO FINAL 1: Agente Inteligente con RAG y Herramientas Externas
-------------------------------------------------------------------
En este proyecto final, vamos a crear un agente chatbot avanzado que combina:
1. MEMORIA: Para recordar el contexto de la conversación.
2. RAG (Pinecone): Una base de datos vectorial con nuestros propios documentos.
3. HERRAMIENTAS EXTERNAS: Capacidad de buscar en Wikipedia o usar herramientas matemáticas.

EL OBJETIVO:
El agente debe razonar y decidir qué herramienta usar:
- Si la pregunta es sobre nuestros documentos internos, usará Pinecone.
- Si la pregunta es sobre cultura general, usará Wikipedia.
- Si es un cálculo, usará la calculadora.

PASOS PARA EL ESTUDIANTE:
1. Asegúrate de tener el archivo "Historia de Espana.pdf" en esta carpeta.
2. Ejecuta el archivo "seed_db.py" para subir la información a Pinecone:
   > python "Leccion 22 (Proyecto Agente 1)/seed_db.py"
3. Una vez subido y procesado, ya puedes ejecutar este archivo "main.py".

Este flujo demuestra cómo un LLM puede actuar como el "cerebro" que coordina diferentes
fuentes de información de manera inteligente.
"""

import os
from dotenv import load_dotenv

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import Tool # Importación corregida para versiones actuales
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Componentes para Pinecone (Como vimos en la Lección 9)
from pinecone import Pinecone

# 1. CONFIGURACIÓN DEL ENTORNO Y MODELO
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings_model = OpenAIEmbeddings()

# 2. CONFIGURACIÓN DEL MÓDULO DE MEMORIA
# Usamos 'chat_history' como clave para que el agente reconozca el historial
memoria = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# 3. CONEXIÓN A PINECONE (RAG - Visto en Lección 9)
print("Conectando con la base de datos de Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchain-curso" # El índice que creamos en la Lección 9
index = pc.Index(index_name)

# 4. CREACIÓN DE LA HERRAMIENTA PERSONALIZADA (CUSTOM TOOL)
def consulta_interna(query: str):
    """
    Esta función busca información en Pinecone y devuelve el texto más relevante.
    """
    # Convertimos la pregunta en un vector (Embedding)
    query_vector = embeddings_model.embed_query(query)
    
    # Buscamos en Pinecone
    busqueda = index.query(vector=query_vector, top_k=1, include_metadata=True)
    
    if busqueda["matches"]:
        # Retornamos el texto guardado en la metadata (como hicimos en la lección 9)
        return busqueda["matches"][0]["metadata"]["text"]
    else:
        return "No se encontró información relevante en los documentos internos."

# Definimos el objeto Tool con su Docstring (Crucial para el razonamiento del agente)
herramienta_rag = Tool(
    name="Consulta interna",
    func=consulta_interna,
    description=(
       "Base de datos de conocimiento técnico y exclusivo sobre Historia de España. "
        "Úsala como fuente de VERDAD ABSOLUTA para nombres, fechas y eventos históricos en España. "
        "Si el usuario pregunta por familias, reyes o sucesos españoles, esta es SIEMPRE tu primera opción. "
        "No busques en Wikipedia si esta herramienta devuelve una respuesta válida."
    )
)

# 5. CONFIGURACIÓN DE HERRAMIENTAS GLOBALES
# Cargamos Wikipedia y Math (Calculadora)
# Wikipedia nos dará información general de otros países o temas actuales
herramientas_base = load_tools(["wikipedia", "llm-math"], llm=llm)

# Combinamos todas las herramientas en un solo set
todas_las_herramientas = herramientas_base + [herramienta_rag]

# 6. INICIALIZACIÓN DEL AGENTE
# Utilizamos CONVERSATIONAL_REACT_DESCRIPTION para que soporte la memoria
agente = initialize_agent(
    tools=todas_las_herramientas,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memoria,
    verbose=True,
    handle_parsing_errors=True
)

# 7. EJECUCIÓN Y PRUEBAS
if __name__ == "__main__":
    print("\n--- ¡Agente Inicializado y listo para el Proyecto Final! ---\n")

    # Pregunta 1: Consulta interna (Historia de España - Debería ir a Pinecone)
    print("PREGUNTA 1: La familia Saez Soriano es una de las más importantes de la historia de España, ¿Quienes son los miembros de la familia Saez Soriano?")
    agente.run(input="La familia Saez Soriano es una de las más importantes de la historia de España, ¿Quienes son los miembros de la familia Saez Soriano?")

    print("\n" + "-"*50 + "\n")

    # Pregunta 2: Uso de Memoria + Conocimiento LLM (Otras regiones)
    print("PREGUNTA 2: ¿Qué pasó durante la misma etapa en Francia?")
    agente.run(input="¿Qué pasó durante la misma etapa en Francia?")

    print("\n" + "-"*50 + "\n")

    # Pregunta 3: Wikipedia (Información Actualizada o General)
    print("PREGUNTA 3: ¿Cuáles son las marcas de vehículos más famosas hoy en día?")
    agente.run(input="¿Cuáles son las marcas de vehículos más famosas hoy en día?")