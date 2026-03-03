"""
LECCIÓN 15: Introducción a los Agentes (El Patrón ReAct)
-------------------------------------------------------
Hasta ahora, nuestras cadenas tenían pasos fijos. Un AGENTE, en cambio, 
es un sistema que usa un LLM como "cerebro" para decidir qué pasos dar 
y en qué orden, utilizando herramientas externas.

¿QUÉ ES EL PATRÓN ReAct (Reason + Act)?
Es la lógica que siguen la mayoría de agentes:
1. Pensamiento (Thought): El modelo razona sobre qué necesita hacer.
2. Acción (Action): El modelo decide qué herramienta usar.
3. Observación (Observation): El modelo ve el resultado de la herramienta.

-------------------------------------------------------------------------
PREGUNTAS FRECUENTES PARA ESTUDIANTES:

1. ¿QUÉ OTRAS HERRAMIENTAS (TOOLS) EXISTEN?
Además de "llm-math" (calculadora), LangChain ofrece muchas más:
- "google-search": Permite al agente buscar en internet en tiempo real.
- "wikipedia": Para consultar datos y definiciones precisas.
- "requests_all": Permite al agente interactuar con cualquier API web.
- "arxiv": Para buscar artículos científicos y académicos.
- Custom Tools: ¡Tú mismo puedes crear funciones Python y dárselas!

2. ¿QUÉ SIGNIFICA ZERO_SHOT_REACT_DESCRIPTION?
Es el "algoritmo de pensamiento" del agente:
- ZERO_SHOT: El agente no ha visto ejemplos, decide sobre la marcha.
- REACT: Usa el bucle Pensamiento -> Acción -> Observación.
- DESCRIPTION: El agente elige la herramienta leyendo su descripción de texto.

3. ¿HAY OTROS TIPOS DE AGENTES?
- CONVERSATIONAL_REACT_DESCRIPTION: Para agentes que necesitan MEMORIA.
- CHAT_ZERO_SHOT_REACT_DESCRIPTION: Optimizado para modelos de Chat (GPT-4).
- STRUCTURED_CHAT: Para herramientas que necesitan varios datos de entrada.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DEL LLM
# Para agentes, solemos usar temperature=0 para que sean precisos y sigan el formato.
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, model="gpt-4o")

# 3. CARGA DE HERRAMIENTAS (TOOLS)
# La herramienta "llm-math" permite al modelo hacer cálculos complejos usando Python.
tools = load_tools(["llm-math"], llm=llm)

# -------------------------------------------------------------------------
# MÉTODO 1: initialize_agent (MÉTODO ANTIGUO / LEGACY)
# -------------------------------------------------------------------------
# Este método era muy sencillo pero está siendo sustituido por sistemas 
# más modulares. Lo mantenemos para que los estudiantes lo reconozcan.
print("\n=== MÉTODO 1: initialize_agent (Legacy) ===")

legacy_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, # Nos permite ver el "pensamiento" en la consola
    handle_parsing_errors=True # Crucial para que no explote si el LLM falla el formato
)

# Ejemplo rápido con el método antiguo
print("\nEjecutando consulta matemática con método antiguo...")
resultado_legacy = legacy_agent.run("¿Cuánto es 25 elevado a la potencia de 0.5 y luego multiplicado por 10?")
print(f"Resultado Final (Legacy): {resultado_legacy}")


# -------------------------------------------------------------------------
# MÉTODO 2: create_react_agent (MÉTODO MODERNO Y RECOMENDADO)
# -------------------------------------------------------------------------
# Aquí separamos la lógica en dos partes: 
# 1. El Agente (El cerebro que decide).
# 2. El AgentExecutor (El cuerpo que ejecuta las herramientas).
print("\n=== MÉTODO 2: create_react_agent (Moderno) ===")

# Definimos el Prompt que le enseña al modelo a ser un Agente.
# Este formato es SACROSANTO; si cambias los nombres, el agente se rompe.
template = """
Responde a la siguiente pregunta lo mejor que puedas: {input}

Tienes acceso a las siguientes herramientas:
{tools}

Usa el siguiente formato:

Question: la pregunta que debes responder
Thought: siempre debes pensar en qué hacer
Action: la acción a realizar, debe ser una de [{tool_names}]
Action Input: la entrada para la acción
Observation: el resultado de la acción
... (este pensamiento/acción/entrada/observación puede repetirse N veces)
Thought: Ahora sé la respuesta final
Final Answer: la respuesta final a la pregunta original

¡Comienza!

Question: {input}
Thought: {agent_scratchpad}
"""

# Variables del Prompt que inyecta LangChain automáticamente:
# {tools} -> Descripción de las herramientas que cargamos.
# {tool_names} -> Solo los nombres de las herramientas.
# {agent_scratchpad} -> El "borrador" donde el agente guarda su historial de razonamiento.

prompt = PromptTemplate.from_template(template)

# Creamos el Agente (El cerebro)
modern_agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Creamos el Ejecutor (La estructura que lo hace funcionar)
agent_executor = AgentExecutor(
    agent=modern_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

print("\nEjecutando consulta con método moderno...")
respuesta_moderna = agent_executor.invoke({"input": "Si tengo 1500 euros y gasto el 18% en un monitor, ¿cuánto dinero me queda?"})

print("\n--- ANÁLISIS FINAL ---")
print(f"Respuesta final obtenida: {respuesta_moderna['output']}")
print("\n💡 NOTA PARA ESTUDIANTES: El Método 2 es más robusto porque nos permite "
      "\npersonalizar el prompt y escalar a sistemas más complejos como LangGraph.")