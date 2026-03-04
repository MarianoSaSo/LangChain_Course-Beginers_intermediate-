"""
LECCIÓN 17: El Agente Programador (Code Python Agent)
---------------------------------------------------
En esta lección vamos a crear un "Becario de Programación". Un agente que no 
solo "sabe" código, sino que tiene una consola de Python (`PythonREPL`) 
donde puede ESCRIBIR, EJECUTAR y CORREGIR su propio código para resolver problemas.

¿CONCEPTOS CLAVE PARA EL ESTUDIANTE?

1. ¿QUÉ ES PANDAS?:
   Es la librería líder en Python para análisis de datos. Imagina un Excel 
   superpotente dentro de tu código. Su estructura principal es el 'DataFrame' 
   (una tabla con filas y columnas). Se usa para filtrar, agrupar y calcular 
   datos de forma masiva.

2. ¿QUÉ ES PythonREPLTool?:
   REPL significa "Read-Eval-Print Loop". Es una consola de Python viva. 
   Esta herramienta permite que el Agente envíe código Python, lo ejecute en 
   TU máquina y reciba el resultado. ¡El agente puede incluso corregir su 
   propio código si falla!

3. ¿CÓMO SABE DÓNDE ESTÁN LOS ARCHIVOS?:
   El agente trabaja en el "Directorio Activo" (CWD). Si ejecutas este script 
   desde la carpeta 'Leccion 17', el agente buscará cualquier archivo (como el .csv) 
   en esa misma ubicación automáticamente.

OBJETIVOS:
1. Aprender a usar `PythonREPLTool` para ejecutar código en tiempo real.
2. Usar `create_python_agent` de `langchain_experimental`.
3. Analizar datos de un DataFrame (Pandas) usando IA.
4. Entender los riesgos de seguridad al dar ejecución de código a un LLM.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_classic.agents import AgentType # Para este tipo de agente específico

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DEL LLM
# Usamos temperature 0 para que el código generado sea determinista y lógico.
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"), 
    temperature=0, 
    model="gpt-4o"
)

# -------------------------------------------------------------------------
# 3. CREACIÓN DE DATOS DE PRUEBA (Para que el alumno pueda ejecutarlo)
# -------------------------------------------------------------------------
# Si el alumno no tiene el Excel, creamos uno rápido para la práctica.
print("\n--- GENERANDO DATOS DE PRUEBA (ventas.csv) ---")
datos = {
    'Producto': ['Laptop', 'Mouse', 'Monitor', 'Laptop', 'Teclado', 'Monitor'],
    'Linea de Producto': ['Hardware', 'Accesorios', 'Hardware', 'Hardware', 'Accesorios', 'Hardware'],
    'Venta total': [1200, 25, 300, 1150, 45, 310]
}
df_ventas = pd.DataFrame(datos)
df_ventas.to_csv('ventas.csv', index=False)
print("Archivo 'ventas.csv' creado con éxito.")

# -------------------------------------------------------------------------
# 4. CREACIÓN DEL AGENTE PROGRAMADOR
# -------------------------------------------------------------------------
# AgentType.ZERO_SHOT_REACT_DESCRIPTION es el estándar para este toolkit.
# PythonREPLTool es la "consola" que el agente usará.
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

# -------------------------------------------------------------------------
# 5. EJEMPLO 1: LÓGICA PURA Y ALGORITMOS
# -------------------------------------------------------------------------
print("\n=== EJEMPLO 1: Resolución de Algoritmos ===")
pregunta_1 = "Genera una lista de los primeros 10 números primos y calcula su suma."
# El agente va a: 1. Pensar un script. 2. Ejecutarlo en el REPL. 3. Dar el resultado final.
agent_executor.invoke(pregunta_1)

# -------------------------------------------------------------------------
# 6. EJEMPLO 2: ANÁLISIS DE DATOS (PANDAS)
# -------------------------------------------------------------------------
print("\n=== EJEMPLO 2: Análisis de Archivo CSV/Excel ===")

# Le explicamos al agente que hay un archivo llamado 'ventas.csv'
pregunta_2 = (
    "Lee el archivo 'ventas.csv' usando pandas. "
    "Dime cuál es la 'Venta total' agregada por cada 'Linea de Producto'. "
    "Asegúrate de mostrar los resultados finales."
)

agent_executor.invoke(pregunta_2)

# -------------------------------------------------------------------------
# 7. CONSEJOS PARA EL ESTUDIANTE (SEGURIDAD)
# -------------------------------------------------------------------------
"""
⚠️ AVISO DE SEGURIDAD PARA LA CLASE:
Darle a un Agente la herramienta PythonREPLTool es como darle a un extraño
las llaves de tu casa. El agente PUEDE:
1. Borrar archivos del sistema.
2. Leer variables de entorno (claves API).
3. Instalar malware si el prompt es malicioso.

SOLUCIÓN: En producción, estos agentes DEBEN ejecutarse en contenedores 
aislados (Sandbox / Docker) para proteger el servidor.
"""

# -------------------------------------------------------------------------
# 8. DESAFÍO PARA EL ESTUDIANTE
# -------------------------------------------------------------------------
"""
🎯 DESAFÍO FINAL:
Pídele al agente que cree un top 3 de los productos más vendidos y que calcule 
qué porcentaje del total de ventas representan los Hardware.
"""

"""
-------------------------------------------------------------------------
RESUMEN PARA EL PROFESOR:
1. Hemos introducido 'langchain_experimental', donde viven estas herramientas.
2. El alumno ve que el LLM no solo escribe código, sino que lo "testea" a sí mismo.
3. Se enseña el flujo: Carga de Datos -> Procesamiento -> Respuesta.
-------------------------------------------------------------------------
"""