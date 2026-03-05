"""
LECCIÓN 19: LCEL (LangChain Expression Language) - El Operador Pipe
-----------------------------------------------------------------
Hasta ahora hemos usado cadenas simples o agentes. Pero hoy en día, la forma 
estándar de construir en LangChain es mediante LCEL. Es un lenguaje declarativo 
que permite encadenar componentes de forma visual usando el símbolo '|'.

¿POR QUÉ USAR LCEL?
1. Streaming: Soporta que la respuesta llegue poco a poco automáticamente.
2. Async: Soporta ejecución asíncrona sin cambiar código.
3. Paralelismo: Permite ejecutar múltiples pasos a la vez.
4. Trazabilidad: Es mucho más fácil de ver en herramientas como LangSmith.

¿CÓMO FUNCIONA?
Piensa en el símbolo '|' como una tubería (pipe). El resultado de lo que hay a 
la izquierda se inyecta directamente como entrada de lo que hay a la derecha.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DE COMPONENTES
# Usamos un modelo de chat y un parser de texto simple.
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-4o", 
    temperature=0.7
)

# --- ¿QUÉ ES EL StrOutputParser? ---
# Por defecto, el LLM nos devuelve un objeto complejo (AIMessage) que contiene:
# - El texto de respuesta ('content')
# - Metadatos (tokens usados, tiempo, etc.)
# El 'StrOutputParser' actúa como un filtro: toma ese objeto complejo y 
# extrae SOLO el texto de respuesta como un String de Python limpio.
# Esto es esencial para que la salida de esta cadena pueda entrar directamente 
# en el siguiente paso de nuestra tubería (|).
parser = StrOutputParser()

# 3. CREACIÓN DEL PROMPT
prompt = ChatPromptTemplate.from_template(
    "Eres un profesor experto en tecnología. Explica de forma breve y sencilla qué es {tema}."
)

# -------------------------------------------------------------------------
# 4. LA CADENA LCEL (EL MOMENTO MÁGICO)
# -------------------------------------------------------------------------
# En lugar de usar LLMChain (Legacy), unimos las piezas con el pipe '|'
# Flujo: Prompt -> LLM -> Parser (Limpieza de texto)
chain = prompt | llm | parser

print("\n=== EJECUTANDO CADENA LCEL ===")
tema_interes = "el Internet de las Cosas (IoT)"
respuesta = chain.invoke({"tema": tema_interes})

print(f"\nExplicación del profesor:\n{respuesta}")

# -------------------------------------------------------------------------
# 5. LCEL AVANZADO: RUNNABLES Y COMBINACIONES
# -------------------------------------------------------------------------
# Podemos hacer cosas más complejas, como pedirle que después de explicarlo, 
# lo traduzca a otro idioma o lo resuma.

traductor_prompt = ChatPromptTemplate.from_template(
    "Traduce el siguiente texto al inglés de forma profesional:\n\n{texto}"
)

# Creamos una cadena que usa el resultado de la anterior
# Nota: Usamos una función lambda para pasar el resultado correctamente.
chain_completa = (
    {"texto": chain}  # El resultado de la primera cadena se mete en 'texto'
    | traductor_prompt 
    | llm 
    | parser
)

print("\n=== EJECUTANDO CADENA COMPUESTA (Explicación + Traducción) ===")
resultado_final = chain_completa.invoke({"tema": "Blockchain"})
print(f"\nResultado en Inglés:\n{resultado_final}")

# -------------------------------------------------------------------------
# 6. DESAFÍO PARA EL ESTUDIANTE
# -------------------------------------------------------------------------
"""
🎯 DESAFÍO:
Crea una cadena LCEL que reciba un nombre de un animal y:
1. Escriba un dato curioso sobre ese animal.
2. Luego, pase ese dato a otro prompt que lo convierta en un poema de 4 versos.
"""

"""
-------------------------------------------------------------------------
RESUMEN PARA EL PROFESOR:
1. Introducimos el concepto de 'Pipe' (|).
2. Dejamos atrás las clases pesadas como LLMChain para usar componentes atómicos.
3. Se enseña cómo pasar diccionarios para alimentar prompts posteriores.
-------------------------------------------------------------------------

🗺️ FLUJO VISUAL DE LA LÍNEA DE MONTAJE (Para tus alumnos):

Imagina que 'chain_completa' es una fábrica con varias estaciones:

1. ENTRADA (Input):
   Usuario envía {"tema": "Blockchain"}

2. ESTACIÓN 1 (La primera cadena 'chain'):
   - Input: "Blockchain"
   - Proceso: Genera la explicación en español.
   - Output: "El Blockchain es una tecnología de registro..."

3. TRANSFERENCIA (El diccionario {"texto": chain}):
   - Aquí LangChain dice: "Toma el resultado anterior y ponle una etiqueta que diga 'texto'".
   - Ahora tenemos: {"texto": "El Blockchain es una tecnología..."}

4. ESTACIÓN 2 (El Traductor Prompt):
   - Lee la etiqueta 'texto' y monta la instrucción: 
     "Traduce esto al inglés: El Blockchain es una tecnología..."

5. ESTACIÓN 3 (El LLM de nuevo):
   - Procesa la instrucción y traduce el contenido.

6. SALIDA FINAL (El Parser):
   - Abre el paquete del LLM, quita los metadatos y entrega el String limpio.

¡Y así es como conectamos piezas como si fueran piezas de LEGO! 🧱
"""
