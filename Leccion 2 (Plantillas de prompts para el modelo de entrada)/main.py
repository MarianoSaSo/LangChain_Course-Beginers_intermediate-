"""
LECCION 2: Plantillas de Prompts (PromptTemplates)
--------------------------------------------------
En esta lección aprenderemos a crear esqueletos de mensajes reutilizables
utilizando variables dinámicas entre llaves {}. 
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    PromptTemplate
)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", temperature=0.7)

# 2. DEFINICIÓN DE PLANTILLAS (Blueprints)
# Definimos el comportamiento del sistema usando una variable {pokemon_especifico}
system_template = "Eres una IA especializada en pokemon de cualquier tipo {pokemon_especifico}. Tu tarea es responder preguntas sobre este pokemon especifico de manera precisa y detallada."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Definimos la estructura de la consulta del usuario
human_template = "Necesito un articulo sobre el pokemon {pokemon_especifico}. El articulo debe incluir su historia, habilidades y características, pero tiene que tener un maximo de 200 palabras"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 3. COMPOSICIÓN DEL CHAT PROMPT
# Combinamos ambas plantillas en un único objeto de conversación
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 4. LÓGICA DE FORMATEO
# Esta función se encarga de inyectar los datos reales en las plantillas
def format_chat(pokemon_especifico_entrada): 
    # Formateamos la plantilla con el nombre del pokemon
    chat_prompt.format_prompt(pokemon_especifico=pokemon_especifico_entrada)
    
    # Convertimos la plantilla ya rellena a una lista de mensajes (SystemMessage, HumanMessage)
    # que es el formato que el modelo de lenguaje (LLM) espera recibir.
    solicitud_completa = chat_prompt.format_prompt(pokemon_especifico=pokemon_especifico_entrada).to_messages()
    return solicitud_completa 


# 5. EJECUCIÓN Y LLAMADA AL MODELO
print(f"--- Generando contenido para: Squirtle ---")
mensajes = format_chat("Squirtle")
respuesta = llm.invoke(mensajes)

# 6. SALIDA DE RESULTADOS
print(respuesta.content)