"""
LECCIÓN 18: Creación de Herramientas Personalizadas (Custom Tools)
-----------------------------------------------------------------
En esta lección vamos a aprender que el verdadero poder de un Agente no es solo 
usar lo que ya viene en LangChain, sino CONECTARLO con nuestras propias funciones.

¿EL SECRETO? EL DOCSTRING.
Para un Agente, el "docstring" (la descripción de la función) no es solo documentación 
para humanos; es el MANUAL DE INSTRUCCIONES que el Agente lee para decidir si 
usar la herramienta o no.

CONCEPTOS CLAVE:
1. @tool: El decorador que convierte una función normal de Python en una herramienta.
2. Docstring Dinámico: Debe ser descriptivo. Si el LLM no entiende para qué sirve 
   la herramienta, nunca la llamará.
3. Input Schema: El Agente necesita saber qué enviarle a la función (texto, números, etc).
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
# Importamos desde langchain_classic como en las lecciones anteriores
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import tool # La forma estándar y moderna de crear herramientas
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DEL LLM
# Usamos temperature 0 para que el agente sea preciso al elegir herramientas.
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"), 
    temperature=0.0, 
    model="gpt-4o"
)

# -------------------------------------------------------------------------
# 3. CREACIÓN DE HERRAMIENTAS PERSONALIZADAS (CUSTOM TOOLS)
# -------------------------------------------------------------------------
# IMPORTANTE: Observa cómo el docstring explica exactamente CUÁNDO usar la función.

@tool
def persona_amable(text: str) -> str:
    """
    Retorna el nombre de la persona más amable del universo. 
    Útil cuando alguien pregunte por amabilidad o personas bondadosas.
    El argumento de entrada puede estar vacío.
    """
    return "Anibal Saez Soriano"

@tool
def hora_actual(text: str) -> str:
    """
    Retorna la fecha y hora actual del sistema. 
    DEBES usar esta función para cualquier consulta sobre la hora o el momento presente. 
    Para fechas que no sean la hora actual (como fechas históricas), debes usar otra herramienta.
    """
    return str(datetime.now())

# -------------------------------------------------------------------------
# 4. PREPARACIÓN DE HERRAMIENTAS EXTERNAS
# -------------------------------------------------------------------------
# Creamos la herramienta de Wikipedia configurada en español
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="es"))

# Cargamos la herramienta matemática
math_tools = load_tools(["llm-math"], llm=llm)

# -------------------------------------------------------------------------
# 5. UNIÓN DE TODAS LAS HERRAMIENTAS
# -------------------------------------------------------------------------
# Combinamos nuestras funciones con las herramientas de la comunidad.
tools = [wikipedia, persona_amable, hora_actual] + math_tools

# -------------------------------------------------------------------------
# 6. CONFIGURACIÓN DEL AGENTE
# -------------------------------------------------------------------------
# Usamos el modo "Legacy" para esta lección por su sencillez educativa.
print("\n=== INICIALIZANDO AGENTE DE HERRAMIENTAS PERSONALIZADAS ===")

agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)

# -------------------------------------------------------------------------
# 7. PRUEBAS DEL AGENTE
# -------------------------------------------------------------------------

# Prueba 1: Consulta sobre la hora (usará hora_actual)
print("\n--- TEST 1: ¿Qué hora es? ---")
agent.invoke("Dime la hora actual por favor.")

# Prueba 2: Consulta sobre amabilidad (usará persona_amable)
print("\n--- TEST 2: La persona más amable ---")
agent.invoke("¿Quién es la persona más amable del mundo?")

# Prueba 3: Combinación de herramientas
print("\n--- TEST 3: Multitarea ---")
agent.invoke("Dime quién es la persona más amable y luego busca en wikipedia quién fue Albert Einstein.")

# -------------------------------------------------------------------------
# 8. DESAFÍO PARA EL ESTUDIANTE
# -------------------------------------------------------------------------
"""
🎯 DESAFÍO FINAL:
Crea una nueva herramienta llamada 'calculadora_propina' que reciba un monto 
(como string) y devuelva el 10% de ese monto como propina sugerida.
¡No olvides redactar un docstring claro para que el agente sepa usarla!
"""

"""
-------------------------------------------------------------------------
RESUMEN PARA EL PROFESOR:
1. Se refuerza que el LLM no "adivina", sino que lee los docstrings.
2. Se muestra cómo mezclar herramientas de sistema con funciones de Python puro.
3. Se mantiene el estilo de 'initialize_agent' para facilitar la lectura al alumno.
-------------------------------------------------------------------------
"""
