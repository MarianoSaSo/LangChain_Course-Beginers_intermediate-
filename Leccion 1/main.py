"""
LECCION 1: Fundamentos de LangChain y Modelos de Chat
-----------------------------------------------------
En esta lección aprenderemos a configurar el modelo, cargar variables de entorno
y utilizar los diferentes tipos de mensajes (System y Human).
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. CONFIGURACIÓN DEL ENTORNO
# Usamos find_dotenv() para localizar el archivo .env en la raíz del proyecto
load_dotenv(find_dotenv())

# 2. INICIALIZACIÓN DEL MODELO
# Configuramos el modelo, la temperatura (creatividad) y cargamos la API Key desde el entorno
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.7)


print("--- EJEMPLO 1: Consulta Simple ---")
# La Opción 1 es ideal para preguntas rápidas tipo 'pregunta-respuesta'.
respuesta_simple = llm.invoke([
    HumanMessage(content="¿Cuál es el pokemon número 6?")
])
print(f"Respuesta: {respuesta_simple.content}\n")


print("--- EJEMPLO 2: Conversación con Contexto ---")
# La Opción 2 nos permite definir una 'personalidad' (SystemMessage)
# y llevar un historial o contexto de la conversación.
mensajes = [
    SystemMessage(content="Eres un experto profesor de Pokémon. Respondes de forma breve y emocionante."),
    HumanMessage(content="¿Cuál es el pokemon más poderoso de todos?")
]

respuesta_contexto = llm.invoke(mensajes)
print(f"Respuesta del profesor: {respuesta_contexto.content}")