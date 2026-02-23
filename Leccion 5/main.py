from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
import os
from dotenv import load_dotenv

from fastapi import FastAPI


# 1. Cargar variables de entorno
load_dotenv()

# 2. Crear modelo
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 3. Función para buscar artículos y generar respuesta
def get_wikipedia_articles(persona, pregunta):
    # Buscar artículos en Wikipedia
    docs = WikipediaLoader(query=persona, load_max_docs=3, lang="es").load()
    contexto_extra = docs[0].page_content if docs else "No se encontraron artículos relevantes en Wikipedia."

    # Crear el prompt humano
    human_prompt = HumanMessagePromptTemplate.from_template(
        "Eres un experto en historia y te han preguntado: {pregunta}. Usa el contenido de Wikipedia obtenido en \"{contenido}\" para responder de manera precisa y completa."
    )

    # Crear el chat prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("Eres un experto en historia."),
            human_prompt,
        ]
    )

    # Formatear mensajes
    mensajes = chat_prompt.format_messages(pregunta=pregunta, contenido=contexto_extra)

    # Obtener respuesta del modelo
    respuesta = chat.invoke(mensajes)

    # Imprimir resultado
    print("✅ Respuesta generada:\n")
    print(respuesta.content)

# Ejemplo de uso:
get_wikipedia_articles("Stephen King", "¿Quien es?")


# CREACION DE PRUEBA PARA POSTMAN CON FASTAPI
#@¬@¬@¬@¬@¬@¬¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬@¬
#Crear la aplicación FastAPI
app = FastAPI()
# Definir el modelo de solicitud
class RequestModel(BaseModel):
    persona: str
    pregunta: str   
# Definir el endpoint
@app.post("/ask")   
def ask_wikipedia(request: RequestModel):
    get_wikipedia_articles(request.persona, request.pregunta)
    return {"message": "Consulta procesada correctamente."}
