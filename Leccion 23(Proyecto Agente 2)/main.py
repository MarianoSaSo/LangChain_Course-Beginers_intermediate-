"""
PROYECTO FINAL 2: Agente SQL - Eliminando la Barrera del Lenguaje Técnico
------------------------------------------------------------------------
En esta lección final, alcanzamos uno de los hitos más potentes de LangChain: 
la capacidad de convertir un LLM en un experto en Bases de Datos SQL.

¿CUÁL ES EL PROBLEMA QUE RESOLVEMOS?
Tradicionalmente, para analizar datos en una empresa, necesitabas saber SQL. 
Con este Agente, cualquier usuario puede preguntar en lenguaje natural (español)
y la IA se encargará de:
1. ENTENDER la estructura de tus tablas (Schema Awareness).
2. TRADUCIR el lenguaje natural a una consulta SQL válida.
3. EJECUTAR la consulta y "humanizar" el resultado.

REQUISITOS PREVIOS PARA ESTUDIANTES:
1. Instalar MySQL Community Server (Versión Full).
2. Configurar la base de datos 'world' usando el script 'setup_world_db.sql'.
3. Tener el archivo 'password_sql.txt' con la clave '123456' en la carpeta raíz.
4. Instalar dependencias: pip install mysql-connector-python sqlalchemy
"""

import os
from dotenv import load_dotenv
import mysql.connector

# Importaciones de LangChain
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

# 1. SETUP DE ENTORNO Y MODELO
load_dotenv()
# Usamos gpt-4o por su excelente capacidad para generar código SQL preciso
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. GESTIÓN DE SEGURIDAD (PASSWORD)
# IMPORTANTE: Nunca escribas contraseñas directamente en el código (Hardcoding).
# La lección enseña a leerla de un archivo externo por seguridad.
try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_password = os.path.join(base_dir, "password_sql.txt")
    with open(path_password, "r") as f:
        pass_SQL = f.read().strip()
except FileNotFoundError:
    print(f"⚠️ Alerta: No se encontró 'password_sql.txt' en {path_password}")
    pass_SQL = "" 

# Configuración centralizada de la base de datos
db_config = {
    "user": "root",
    "password": pass_SQL,
    "host": "localhost",
    "database": "world",
    "port": 3306
}

# -------------------------------------------------------------------------
# MÉTODO A: LA FORMA TRADICIONAL (MySQL Connector)
# -------------------------------------------------------------------------
# Mostramos esto para que el alumno aprecie la diferencia entre 
# programar una consulta fija y tener un agente inteligente.
print("\n=== MÉTODO A: EJECUCIÓN MANUAL (Código Rígido) ===")
try:
    conexion = mysql.connector.connect(**db_config)
    cursor = conexion.cursor()

    query = "SELECT SUM(Population) FROM country WHERE Continent = 'Asia';"
    cursor.execute(query)
    resultado = cursor.fetchone()
    
    print(f"Resultado (SQL Puro): La población calculada en Asia es {resultado[0]}")
    
    cursor.close()
    conexion.close()
except Exception as e:
    print(f"Error en conexión manual: {e}")


# -------------------------------------------------------------------------
# MÉTODO B: EL AGENTE SQL DE LANGCHAIN (La Magia)
# -------------------------------------------------------------------------
print("\n=== MÉTODO B: AGENTE INTELIGENTE (Lenguaje Natural) ===")

# 1. Definimos el URI de conexión (Formato estándar para SQLAlchemy)
# mysql+mysqlconnector://usuario:contraseña@servidor/base_de_datos
uri = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"

# 2. Creamos el objeto Database (Permite al LLM 'ver' las tablas)
db = SQLDatabase.from_uri(uri)

# 3. Inicializamos el Agente
# verbose=True: Fundamental para ver el razonamiento y la query SQL generada.
agente_sql = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True
)

# --- ESCENARIOS DE PRUEBA PARA LA CLASE ---

# ESCENARIO 1: Consulta Directa
# El agente debe identificar las columnas 'Population' y 'Continent' automáticamente.
print("\n[PRUEBA 1]: Población total de Asia")
agente_sql.run("Dime la población total de Asia")

print("\n" + "-"*60)

# ESCENARIO 2: Análisis Complejo (Agregaciones y Ordenación)
# Aquí el agente debe realizar un promedio (AVG), agrupar por región (GROUP BY) 
# y ordenar (ORDER BY) sin que le digamos los comandos SQL.
print("\n[PRUEBA 2]: Análisis de Esperanza de Vida")
pregunta_compleja = (
    "Haz un análisis del promedio de la esperanza de vida por región. "
    "Ordénalo de mayor a menor y dame las 5 mejores."
)
agente_sql.run(pregunta_compleja)

"""
🎯 NOTA FINAL PARA EL ESTUDIANTE:
Fíjate en la consola cómo el agente primero consulta el esquema de las tablas 
('Action: sql_db_schema') antes de escribir la pregunta. Esto es lo que permite 
que el agente sea dinámico y funcione con cualquier base de datos que le des.
"""
