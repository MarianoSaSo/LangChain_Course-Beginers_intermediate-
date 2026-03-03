"""
LECCIÓN 11: Memoria Automática con ConversationBufferMemory
------------------------------------------------------------
En la lección anterior aprendimos a usar ChatMessageHistory de forma manual. 
Hoy daremos un salto hacia la AUTOMATIZACIÓN con ConversationBufferMemory.

¿QUÉ ES CONVERSATIONBUFFERMEMORY?
Es un objeto de memoria avanzado. A diferencia del historial manual, este objeto 
está diseñado para trabajar en equipo con una "Chain" (Cadena). 

LOS ATRIBUTOS CLAVE (Respondiendo a dudas comunes):

1. memory_key="history":
   La cadena (ConversationChain) tiene una plantilla interna (prompt) con un 
   espacio reservado para el historial. Ese espacio tiene el nombre "history". 
   Si llamamos a nuestra llave de memoria de otra forma, la cadena no sabrá 
   dónde insertar los mensajes previos. "history" es el estándar por defecto.

2. return_messages=True:
   Los modelos de chat modernos no quieren recibir un bloque de texto plano. 
   Quieren una lista estructurada de objetos (HumanMessage, AIMessage). 
   Este parámetro le dice a la memoria: "No me des un string, dame la lista de objetos".

FLUJO AUTOMÁTICO:
Cuando usamos .predict(), la cadena hace 3 pasos invisibles:
  A) Guarda tu pregunta actual en la memoria.
  B) Consulta la memoria, junta todo el historial y se lo envía al modelo.
  C) Recibe la respuesta del modelo y la guarda automáticamente en la memoria.
"""

import os
import pickle
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# Usamos langchain_classic para entornos con esa configuración de compatibilidad
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DEL MODELO
# Usamos gpt-4o para mejores respuestas en español
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# 3. CREACIÓN DEL OBJETO DE MEMORIA
# Aquí es donde ocurre la magia: configuramos la 'etiqueta' y el formato de salida.
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 4. CREACIÓN DE LA CADENA CONVERSACIONAL
# verbose=True es fundamental para aprender: nos mostrará en verde cómo LanChain
# construye el prompt mezclando el historial con la pregunta actual.
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

print("--- INICIO DE LA CONVERSACIÓN ---")

# 5. INTERACCIÓN AUTOMATIZADA
# Al usar .predict(), ya no tenemos que gestionar 'history.add_user_message'
# El objeto 'memory' se encarga de todo el ciclo de vida del mensaje.
answer1 = conversation.predict(input="Hola, ¿Cuál es la capital de la Región de Murcia?")
print(f"\nRespuesta 1: {answer1}")

answer2 = conversation.predict(input="Dime además cuáles son los 5 pueblos más importantes de esa región.")
print(f"\nRespuesta 2: {answer2}")

# 6. INSPECCIÓN DE LA MEMORIA ("Bajo el capó")
# Accedemos a memory.buffer para ver cómo están guardados los objetos.
print("\n--- ESTADO INTERNO DEL BUFFER ---")
for message in memory.buffer:
    role = "Usuario" if isinstance(message, HumanMessage) else "IA"
    print(f"{role}: {message.content}")

# 7. PERSISTENCIA: ¿CÓMO GUARDAR EL CEREBRO DE LA IA?
# La memoria solo vive en la RAM. Si cerramos el script, se borra.
# Usamos 'pickle' para serializar (congelar) el objeto y guardarlo en un archivo.
print("\n--- GUARDANDO MEMORIA EN DISCO (.pkl) ---")
with open("conversation_memory.pkl", "wb") as f:
    # Guardamos el objeto entero 'conversation.memory'
    pickle.dump(conversation.memory, f)
print("Archivo 'conversation_memory.pkl' guardado.")

# 8. CARGA DE MEMORIA: RECUPERAR EL CONTEXTO
# Imaginemos que esta es una ejecución nueva en otro momento.
print("\n--- RECARGANDO MEMORIA Y CONTINUANDO ---")
with open("conversation_memory.pkl", "rb") as f:
    memory_reloaded = pickle.load(f)

# Creamos una nueva cadena dándole la memoria que acabamos de cargar
new_conversation = ConversationChain(
    llm=chat, 
    memory=memory_reloaded, 
    verbose=False 
)

print("\nVerificación de que recuerda lo anterior:")
for message in new_conversation.memory.buffer:
    role = "Usuario" if isinstance(message, HumanMessage) else "IA"
    print(f"{role}: {message.content}")

"""
💡 NOTA PARA EL FUTURO: GESTIÓN DE MÚLTIPLES USUARIOS
---------------------------------------------------
En este ejemplo, la memoria es global y se guarda en un solo archivo. 
Pero, ¿qué pasa si tenemos 100 usuarios en pestañas diferentes? 

Para escalar esto en el mundo real:
1. Usamos un 'session_id' (un DNI único para cada chat).
2. En lugar de guardar un archivo local, conectamos esta memoria a una 
   base de datos (como Redis o SQLite).
3. LangChain buscará automáticamente el historial que coincida con el 
   'session_id' que envíe el frontend, evitando que los mensajes de un 
   usuario se mezclen con los de otro.
"""


