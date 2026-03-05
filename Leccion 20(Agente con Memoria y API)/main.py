"""
LECCIÓN 20: Agentes con Memoria y Despliegue con FastAPI
-------------------------------------------------------
En esta lección vamos a dar dos pasos gigantes:
1. MEMORIA: Haremos que nuestro agente recuerde lo que dijimos antes.
2. API: Expondremos nuestro agente al mundo real usando FastAPI.

¿POR QUÉ NO USAMOS EL MODO LEGACY?
Aunque initialize_agent es fácil, el método moderno (create_react_agent) 
nos permite tener un control TOTAL sobre el prompt. Esto es vital para 
explicarle al agente cómo debe manejar la memoria (chat_history).

CONCEPTOS CLAVE:
1. ConversationBufferMemory: El "almacén" de los mensajes anteriores.
2. chat_history: La variable que inyectamos en el prompt con el historial.
3. FastAPI: El framework más moderno y rápido para crear APIs con Python.
4. Pydantic: Para validar que los datos que entran a nuestra API son correctos.
"""

import os
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

# 2. INICIALIZACIÓN DEL MODELO Y HERRAMIENTAS
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Herramientas
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="es"))
math_tools = load_tools(["llm-math"], llm=llm)
tools = [wikipedia] + math_tools

# -------------------------------------------------------------------------
# 3. MEMORIA (El corazón del Chatbot)
# -------------------------------------------------------------------------
# memory_key="chat_history" -> Es el nombre que usaremos en el prompt.
# return_messages=True -> IMPORTANTE: Para que guarde objetos de mensaje, no solo texto.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------------------------------------------------------
# 4. EL PROMPT MODERNO (ReAct con Memoria) - EXPLICACIÓN DEL CABLEADO
# -------------------------------------------------------------------------
# Para el estudiante: ¿Cómo sabe LangChain dónde meter cada cosa? 
# El AgentExecutor actúa como un "administrador" que rellena estos huecos:
#
# - {chat_history}: El Executor va a la 'memory', saca los mensajes y los pega aquí.
# - {tools}: El Executor lee los nombres y descripciones de tu lista 'tools' y los pega aquí.
# - {tool_names}: Una simple lista de los nombres de las herramientas.
# - {input}: Es la pregunta que el usuario envía por la API.
#
# - {agent_scratchpad}: (EL GRAN MISTERIO) 
#   Es el "bloc de notas" o "borrador" del agente. Cuando el agente decide usar una 
#   herramienta, el resultado de esa herramienta NO se manda al usuario directamente, 
#   sino que se escribe en el 'scratchpad'. Así, el agente vuelve a leer el prompt y ve:
#   "Ah, ya usé la calculadora y el resultado fue 10, ahora puedo dar la respuesta final".
#   SIN ESTA LLAVE, EL AGENTE NO TIENE MEMORIA DE CORTO PLAZO Y SE QUEDARÍA EN BUCLE.

template = """
Eres un asistente virtual muy servicial que tiene acceso a herramientas externas.
Siempre intentas usar el contexto de la conversación anterior para ayudar al usuario.

HISTORIAL DE CONVERSACIÓN:
{chat_history}

HERRAMIENTAS DISPONIBLES:
{tools}

FORMATO DE RESPUESTA:
Para responder, debes seguir estrictamente este formato:

Thought: ¿Qué ha dicho el usuario antes? ¿Necesito una herramienta?
Action: la herramienta a usar (debe ser una de [{tool_names}])
Action Input: la entrada para la herramienta
Observation: el resultado de la herramienta
... (este proceso puede repetirse)
Thought: Ya tengo toda la información o no necesito herramientas.
Final Answer: tu respuesta final al usuario basada en el historial y las herramientas.

PREGUNTA DEL USUARIO: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# 5. CREACIÓN DEL AGENTE (MODERNO)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# El Ejecutor gestiona el bucle de pensamiento y la memoria
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# -------------------------------------------------------------------------
# 6. API CON FASTAPI
# -------------------------------------------------------------------------
app = FastAPI(
    title="Mi Primer Agente con Cerebro y API",
    description="Lección 20: Un agente que recuerda y se sirve vía HTTP."
)

# Definimos cómo debe ser la "caja" de datos que llega a la API
class Peticion(BaseModel):
    pregunta: str

@app.get("/")
def home():
    return {"mensaje": "El agente está vivo. Envía un POST a /chat para hablar."}

@app.post("/chat")
def chat(peticion: Peticion):
    # Lógica de salida
    if peticion.pregunta.lower() == "salir":
        return {"respuesta": "¡Adiós! Cerrando sesión de memoria..."}
    
    # Invocamos al agente
    # Nota: No hace falta pasarle el chat_history, AgentExecutor lo saca de la memoria solo.
    resultado = agent_executor.invoke({"input": peticion.pregunta})
    
    return {
        "respuesta": resultado["output"],
        "estado": "procesado"
    }

# -------------------------------------------------------------------------
# 7. EJECUCIÓN DEL SERVIDOR
# -------------------------------------------------------------------------
# Para ejecutar esta lección: 
# Opción A: Ejecuta directamente este archivo (python main.py)
# Opción B: Usa el comando: uvicorn main:app --reload

if __name__ == "__main__":
    print("\n🚀 Servidor levantado en http://127.0.0.1:8000")
    print("💡 TIP: Usa Postman o Thunder Client para enviar un POST a http://127.0.0.1:8000/chat")
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""
🎯 DESAFÍO PARA EL ESTUDIANTE:
1. Intenta preguntar al agente tu nombre y luego pregúntale "¿Cómo me llamo?" 
   para verificar que la memoria funciona.
2. Añade una nueva herramienta personalizada (como la hora actual de la Lección 18) 
   y verifica que el agente puede usarla a través de la API.
"""
