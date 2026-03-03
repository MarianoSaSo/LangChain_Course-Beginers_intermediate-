"""
LECCIÓN 12: Memoria con Ventana Deslizante (ConversationBufferWindowMemory)
--------------------------------------------------------------------------
En las lecciones anteriores vimos memorias que guardaban TODO. Pero en 
proyectos reales, enviar miles de mensajes en cada pregunta es CARO y LENTO.

¿QUÉ ES CONVERSATIONBUFFERWINDOWMEMORY?
Es una variante de la memoria que solo mantiene una "ventana" de las últimas 
interacciones. Cuando llega un mensaje nuevo y se supera el límite, el mensaje 
más antiguo se "olvida".

EL PARÁMETRO CLAVE: k
- 'k' representa el número de INTERACCIONES completas (Pregunta + Respuesta) 
  que la memoria debe recordar.
- IMPORTANTE: Si pones k=2, recordará los últimos 4 mensajes (2 Humanos + 2 IA).

VENTAJAS:
1. Ahorro de Tokens: El tamaño del prompt no crece infinitamente.
2. Rendimiento: El modelo responde más rápido al tener menos contexto que leer.

INCONVENIENTES:
- Pérdida de Contexto: Si la conversación es muy larga, la IA olvidará 
  detalles mencionados al principio (como el nombre del usuario).
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
# Usamos langchain_classic para mantener la compatibilidad con tu entorno
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationChain

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DEL MODELO
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# 3. CREACIÓN DE LA MEMORIA CON VENTANA
# 'k=2': Recordará las últimas 2 interacciones (4 mensajes en total).
# 'memory_key="history"': Lo ponemos de forma explícita para que la cadena sepa
# exactamente dónde buscar el historial, igual que hicimos en la Lección 11.
memory = ConversationBufferWindowMemory(memory_key="history", k=2, return_messages=True)

# 4. CREACIÓN DE LA CADENA
# El modo verbose=True nos permitirá ver cómo LangChain solo envía al 
# modelo los mensajes que entran dentro de la ventana 'k'.
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

print("--- PRUEBA DE VENTANA DESLIZANTE (k=2) ---")

# Vamos a realizar 3 interacciones. Con k=2, la primera interacción 
# ("Hola, me llamo Mariano") debería olvidarse en la tercera pregunta.
print("\nInteracción 1:")
conversation.predict(input="Hola, me llamo Mariano.")

print("\nInteracción 2:")
conversation.predict(input="Vivo en Murcia y me gusta la programación.")

print("\nInteracción 3 (Aquí la IA ya debería haber 'soltado' la Interacción 1):")
answer = conversation.predict(input="¿Sabes cómo me llamo?")
print(f"\nRespuesta de la IA: {answer}")

# 5. COMPROBACIÓN DEL BUFFER
print("\n--- CONTENIDO ACTUAL DE LA MEMORIA ---")
# Veremos que solo hay 4 mensajes (2 pares de Pregunta/Respuesta)
for message in memory.buffer:
    role = "Usuario" if isinstance(message, HumanMessage) else "IA"
    print(f"{role}: {message.content}")