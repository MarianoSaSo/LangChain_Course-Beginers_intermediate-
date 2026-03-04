"""
LECCIÓN 16: Agentes Avanzados (Multi-herramientas y Custom Tools)
---------------------------------------------------------------
En esta lección subimos de nivel. Un agente no solo usa una calculadora; 
puede navegar por internet, consultar Wikipedia y usar funciones CREADAS POR TI
que se conectan con el mundo real mediante APIs externas.

OBJETIVOS:
1. Coordinar múltiples herramientas (Search + Math + Wikipedia).
2. Implementar el Agente con el método MODERNO (AgentExecutor).
3. Aprender a crear herramientas personalizadas (Custom Tools).
4. Conectar un Agente con una API real (CoinGecko) vía Requests.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool # Importante para crear nuestras propias herramientas
import requests # Importamos requests para llamadas a APIs externas

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DEL LLM
# Usamos un modelo potente (gpt-4o) y temperatura 0 para máxima precisión.
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"), 
    temperature=0.0, 
    model="gpt-4o"
)

# -------------------------------------------------------------------------
# 3. CREACIÓN DE UNA HERRAMIENTA PERSONALIZADA (CUSTOM TOOL)
# -------------------------------------------------------------------------
# A veces, el agente necesita algo que no está en internet, como una función
# interna de nuestra empresa o un cálculo muy específico.

@tool
def obtener_precio_cripto(cripto: str) -> str:
    """Útil para obtener el PRECIO REAL de una criptomoneda en USD. 
    Debes pasarle el ID de la moneda (ej: 'bitcoin', 'ethereum' o 'solana')."""
    
    # Limpiamos la entrada (el LLM a veces añade comillas o espacios)
    cripto_id = cripto.strip().replace('"', '').replace("'", "").lower()
    
    # 🔗 CONEXIÓN A API REAL (CoinGecko)
    # Explicación: Usar una API dedicada nos da un dato puro y exacto.
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cripto_id}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if cripto_id in data:
            precio = data[cripto_id]['usd']
            return f"El precio REAL actual de {cripto_id.capitalize()} es ${precio:,} USD."
        else:
            return f"No se encontró información para '{cripto_id}'. Verifica el nombre en inglés."
            
    except Exception as e:
        return f"Error de conexión con la API: {e}"

# -------------------------------------------------------------------------
# 4. CARGA Y COMBINACIÓN DE HERRAMIENTAS
# -------------------------------------------------------------------------

# Herramientas predefinidas de LangChain
# 💡 NOTA: Ya no necesitamos hacer os.getenv("SERPAPI_API_KEY") manualmente.
# Si la variable está en el .env, load_tools la detecta automáticamente por su nombre estándar.
built_in_tools = load_tools(["serpapi", "llm-math", "wikipedia"], llm=llm)

# Combinamos con nuestra nueva herramienta personalizada
tools = built_in_tools + [obtener_precio_cripto]

# -------------------------------------------------------------------------
# 5. CONFIGURACIÓN DEL AGENTE MODERNO
# -------------------------------------------------------------------------
# En la Lección 15 vimos esto, pero ahora lo aplicamos a un caso REAL.
print("\n=== CONFIGURANDO AGENTE MULTI-HERRAMIENTA ===")

# Definimos el Prompt ReAct (el formato SACROSANTO)
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
Final Answer: la respuesta final a la pregunta original en ESPAÑOL.

¡Comienza!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# Construimos el agente (el cerebro)
# 🚀 ¿POR QUÉ NO USAMOS AgentType.ZERO_SHOT_REACT_DESCRIPTION?
# Ese método es 'Legacy' (antiguo). Al usar create_react_agent pasamos un 
# prompt explícito. Esto permite que el estudiante vea y entienda el 
# "proceso de pensamiento" que le estamos pidiendo al modelo.
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Creamos el ejecutor (el cuerpo que ejecuta las herramientas)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True # Crucial para que no explote si el LLM falla el formato
)

# -------------------------------------------------------------------------
# 6. PRUEBA DE FUEGO (MULTITASKING)
# -------------------------------------------------------------------------
print("\n--- INICIANDO CONSULTA COMPLEJA ---")

# Esta pregunta obliga al agente a coordinar herramientas en cadena:
# 1. Wikipedia/Search: Busca el año de nacimiento (Dato Externo).
# 2. LLM-Math: Realiza el cálculo matemático (Lógica de Computación).
# 3. Custom Tool: Consulta nuestra función privada de Cripto (Lógica de Negocio).
consulta = (
    "¿En qué año nació Albert Einstein? "
    "Dime ese año multiplicado por 3. "
    "Y por último, extra: ¿cuál es el precio de bitcoin según tu herramienta personalizada?"
)

resultado = agent_executor.invoke({"input": consulta})

print("\n--- ANÁLISIS FINAL PARA EL ESTUDIANTE ---")
print(f"Respuesta final obtenida: {resultado['output']}")

# -------------------------------------------------------------------------
# 7. DESAFÍO PARA EL ESTUDIANTE (EJERCICIO)
# -------------------------------------------------------------------------
"""
🎯 DESAFÍO:
Crea una nueva herramienta personalizada llamada 'obtener_clima' que reciba 
una ciudad y devuelva un clima inventado (ej: 'Soleado, 25°C').
Luego, intenta que el agente responda: 
'¿Qué clima hace en Madrid y cuánto es la temperatura multiplicada por 2?'
"""

"""
-------------------------------------------------------------------------
RESUMEN TÉCNICO PARA EL PROFESOR:
1. AGENT_SCRATCHPAD: Es el "espacio de trabajo" donde el agente guarda los 
   pasos intermedios. Es vital para que sepa qué ha hecho y qué le falta.
2. TEMPERATURE 0: Crucial en agentes para evitar que el LLM invente nombres 
   de herramientas que no existen o rompa el formato ReAct.
3. HANDLE_PARSING_ERRORS: Los agentes a veces se "emocionan" y no cierran el 
   formato correctamente. Este flag permite que el sistema intente recuperarse.
-------------------------------------------------------------------------
"""
