"""
LECCION 3: Output Parsers y despliegue con FastAPI
--------------------------------------------------
En esta lección aprenderemos a transformar las respuestas de texto de la IA
en estructuras de datos de Python (listas) y a exponer nuestra lógica
a través de una API web profesional.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import CommaSeparatedListOutputParser

from fastapi import FastAPI
from pydantic import BaseModel

# 1. CONFIGURACIÓN DEL MODELO
load_dotenv(find_dotenv())
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# 2. OUTPUT PARSER (Procesador de salida)
# Este objeto le enseña a LangChain cómo convertir un texto separado por comas en una lista []
output_parser = CommaSeparatedListOutputParser()

print("--- DEMOSTRACIÓN: Funcionamiento del Parser ---")
respuesta_imaginaria = "Pikachu, Charmander, Bulbasaur"
respuesta_parseada = output_parser.parse(respuesta_imaginaria)
print(f"1. Texto original (String): {respuesta_imaginaria}")
print(f"2. Resultado procesado (List): {respuesta_parseada}\n")


# 3. PROMPT ENGINEERING CON INSTRUCCIONES DE FORMATO
# Es vital decirle a la IA EXACTAMENTE cómo queremos la separación (comas).
system_template = (
    "Eres un experto en pokemon. El usuario te va a dar un elemento (agua, fuego, etc.) "
    "y tú tienes que responder con una lista de los pokemon que pertenecen a ese elemento. "
    "REGLA CRÍTICA: La respuesta debe ser SOLO una lista separada por comas (ej: A, B, C). "
    "Máximo 10 pokemon y sin numeración."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "Dame una lista de los pokemon que pertenecen al elemento {elemento}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Combinamos las plantillas
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
]) 


# 4. LÓGICA DE NEGOCIO (Formateo y llamada)
def format_chat(elemento_entrada):
    """Prepara los mensajes inyectando el elemento seleccionado."""
    solicitud_completa = chat_prompt.format_prompt(elemento=elemento_entrada).to_messages()
    return solicitud_completa   

# 5. EJECUCIÓN POR CONSOLA
print(f"--- Consultando Pokemons de tipo Agua ---")
mensajes = format_chat("Agua")  
respuesta = chat.invoke(mensajes)
# Aquí ocurre la magia: convertimos el texto de la IA en una lista real de Python
lista_final = output_parser.parse(respuesta.content) 
print(f"Lista final procesada: {lista_final}\n")


# 6. EXTRA: IMPLEMENTACIÓN DE API CON FASTAPI
# Creamos la instancia de la aplicación
app = FastAPI(title="Pokemon AI API", description="API para consultar pokemons por tipo usando LangChain")

# Definimos el esquema de los datos que esperamos recibir (Request Body)
class PokemonRequest(BaseModel):
    elemento: str
    
@app.post("/pokemon/")
def get_pokemon(request: PokemonRequest):
    """Endpoint que recibe un elemento y devuelve datos estructurados."""
    mensajes = format_chat(request.elemento)
    respuesta = chat.invoke(mensajes)
    respuesta_parseada = output_parser.parse(respuesta.content)
    
    return {
        "tipo_consultado": request.elemento,
        "resultados_encontrados": len(respuesta_parseada),
        "pokemons": respuesta_parseada
    }

"""
GUÍA DE PRUEBAS (FastAPI + Uvicorn):
------------------------------------
1. Ejecutar servidor: uvicorn main:app --reload
2. Documentación automática: Abrir http://127.0.0.1:8000/docs en el navegador (¡Es genial!)
3. Probar con Postman o el botón 'Try it out' de la web de docs.
   Body JSON esperado: {"elemento": "fuego"}
"""