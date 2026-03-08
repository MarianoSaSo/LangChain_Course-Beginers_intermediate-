"""
LECCIÓN 21: Agentes con Memoria Persistente (No más amnesia tras reiniciar)
-------------------------------------------------------------------------
En la lección anterior, nuestro agente tenía memoria, pero si apagabas el servidor, 
el agente "moría" y olvidaba todo. Hoy vamos a darle un "disco duro" para que 
el historial sobreviva a los reinicios.

CONCEPTOS CLAVE:
1. Persistencia: Proceso de guardar datos de la RAM (memoria volátil) al Disco (archivo).
2. Serialización (Pickle): Técnica para convertir un objeto de Python (la memoria) 
   en un flujo de bytes que se puede guardar en un archivo .pkl.
3. Ciclo de Vida: Cargar al inicio -> Interactuar -> Guardar al final.

DIFERENCIA CON LA ANTERIOR:
- Lección 20: Memoria Temporal (RAM).
- Lección 21: Memoria Persistente (Disco).
"""

import os
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 1. SETUP DE ENTORNO
load_dotenv()

# Archivo donde guardaremos la memoria
NOMBRE_ARCHIVO_MEMORIA = "memoria_agente.pkl"

# -------------------------------------------------------------------------
# 2. FUNCIONES DE GESTIÓN DE MEMORIA (EL DISCO DURO)
# -------------------------------------------------------------------------

def guardar_memoria(objeto_memoria):
    """Guarda el objeto de memoria de LangChain en un archivo físico."""
    with open(NOMBRE_ARCHIVO_MEMORIA, "wb") as f:
        pickle.dump(objeto_memoria, f)
    print("\n[SISTEMA] Memoria guardada con éxito en disco.")

def cargar_memoria():
    """Carga la memoria desde el archivo si existe; si no, crea una nueva."""
    if os.path.exists(NOMBRE_ARCHIVO_MEMORIA):
        try:
            with open(NOMBRE_ARCHIVO_MEMORIA, "rb") as f:
                memoria_cargada = pickle.load(f)
            print("\n[SISTEMA] Memoria recuperada del archivo.")
            return memoria_cargada
        except Exception as e:
            print(f"\n[ERROR] No se pudo cargar la memoria: {e}. Creando una nueva...")
    
    print("\n[SISTEMA] No hay memoria previa. Iniciando memoria limpia.")
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 3. INICIALIZACIÓN
llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = cargar_memoria()

# Herramientas
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="es"))
math_tools = load_tools(["llm-math"], llm=llm)
tools = [wikipedia] + math_tools

# -------------------------------------------------------------------------
# 4. EL PROMPT (Moderno con Memoria)
# -------------------------------------------------------------------------
template = """
Eres un asistente con memoria persistente. Tus palabras importan porque el 
usuario te recordará incluso si te apagas. Revisa el historial para ser coherente.

HISTORIAL DE CONVERSACIÓN:
{chat_history}

HERRAMIENTAS DISPONIBLES:
{tools}

FORMATO DE RESPUESTA:
Thought: ¿Qué me ha preguntado? ¿Qué herramientas necesito?
Action: la herramienta a usar de [{tool_names}]
Action Input: la entrada para la herramienta
Observation: el resultado de la herramienta
... (repetir si es necesario)
Thought: Ya tengo la respuesta final.
Final Answer: respuesta final al usuario.

PREGUNTA DEL USUARIO: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# 5. CREACIÓN DEL AGENTE Y EJECUTOR
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory, # Cargamos la memoria (nueva o recuperada)
    verbose=True,
    handle_parsing_errors=True
)

# -------------------------------------------------------------------------
# 6. API CON FASTAPI
# -------------------------------------------------------------------------
app = FastAPI(title="Agente con Memoria Persistente")

class Peticion(BaseModel):
    pregunta: str

@app.post("/chat")
def chat(peticion: Peticion):
    pregunta_limpia = peticion.pregunta.lower().strip()

    # Comando especial para apagar y guardar
    if pregunta_limpia == "salir":
        guardar_memoria(agent_executor.memory)
        return {"respuesta": "Memoria guardada. Apagando servidor... ¡Hasta la próxima!"}

    # Ejecutar agente
    resultado = agent_executor.invoke({"input": peticion.pregunta})
    
    # GUARDADO PROACTIVO
    # Guardamos siempre después de cada interacción para no perder nada si se va la luz.
    guardar_memoria(agent_executor.memory)
    
    return {"respuesta": resultado["output"]}

# -------------------------------------------------------------------------
# 7. EJECUCIÓN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n🚀 Servidor en http://127.0.0.1:8000")
    print("Para guardar y salir, envía la palabra 'salir' en el JSON.")
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""
🎯 DESAFÍO PARA EL ESTUDIANTE:
1. Ejecuta el servidor, dile al agente tu animal favorito y apaga el servidor (comando salir).
2. Vuelve a ejecutarlo y pregúntale "¿Cuál es mi animal favorito?". 
3. Analiza el archivo 'memoria_agente.pkl' que se ha creado en la carpeta.
"""
