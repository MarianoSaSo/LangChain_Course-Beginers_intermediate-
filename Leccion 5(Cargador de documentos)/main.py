"""
LECCION 5: Cargadores Dinámicos (Wikipedia) y FastAPI
-----------------------------------------------------
En esta lección aprenderemos a usar cargadores que consultan Internet en tiempo
real y a exponer esta lógica a través de una API para que otras aplicaciones
puedan usar nuestro "Experto en Historia".
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from fastapi import FastAPI
from pydantic import BaseModel

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv(find_dotenv())

# 2. INICIALIZACIÓN DEL MODELO
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# 3. LÓGICA DE NEGOCIO: Consulta a Wikipedia + LLM
def get_wikipedia_articles(persona, pregunta):
    """
    Busca información en Wikipedia sobre una persona y usa un LLM 
    para responder una pregunta específica usando ese contexto.
    """
    # WikipediaLoader: Busca automáticamente en la API de Wikipedia.
    # 'load_max_docs=1' para no saturar el prompt con demasiada información.
    docs = WikipediaLoader(query=persona, load_max_docs=1, lang="es").load()
    
    contexto_extra = docs[0].page_content if docs else "No se encontró información relevante."

    # Definimos la estructura de la conversación
    system_template = SystemMessagePromptTemplate.from_template(
        "Eres un experto en historia que solo responde basado en fuentes fiables."
    )
    human_template = HumanMessagePromptTemplate.from_template(
        "Usa este contenido de Wikipedia: {contenido}\n\nPregunta: {pregunta}"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])

    # Formateamos los mensajes con los datos reales
    mensajes = chat_prompt.format_messages(
        pregunta=pregunta, 
        contenido=contexto_extra
    )

    # Invocamos al modelo
    respuesta = chat.invoke(mensajes)
    return respuesta.content

# --- PRUEBA POR CONSOLA (Solo cuando se ejecuta el archivo directamente) ---
if __name__ == "__main__":
    print("--- EJEMPLO: Consulta sobre Stephen King ---")
    try:
        resultado = get_wikipedia_articles("Stephen King", "¿Quién es y cuál es su obra más famosa?")
        print(f"Respuesta: {resultado}\n")
    except Exception as e:
        print(f"Error al probar la consola: {e}")



# 4. IMPLEMENTACIÓN DE API PROFESIONAL CON FASTAPI
app = FastAPI(
    title="Wikipedia AI Guru",
    description="API que combina Wikipedia con GPT-4o para dar respuestas históricas"
)

# Definimos el esquema de datos (Validación con Pydantic)
class RequestModel(BaseModel):
    persona: str
    pregunta: str   

@app.post("/ask")   
def ask_wikipedia(request: RequestModel):
    """
    Endpoint para realizar consultas dinámicas.
    Body esperado: {"persona": "Marie Curie", "pregunta": "¿Qué descubrió?"}
    """
    respuesta_ia = get_wikipedia_articles(request.persona, request.pregunta)
    
    return {
        "sujeto": request.persona,
        "pregunta": request.pregunta,
        "respuesta_generada": respuesta_ia
    }

# GUÍA DE EJECUCIÓN (FastAPI):
# 1. Ejecutar: uvicorn main:app --reload --port 8001
# 2. Abrir: http://127.0.0.1:8001/docs
