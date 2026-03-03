"""
LECCIÓN 14: Memoria Híbrida - ConversationSummaryBufferMemory
------------------------------------------------------------
Esta es la "joya de la corona" de las memorias en LangChain. Combina la precisión 
de los mensajes literales con la eficiencia del resumen.

¿CÓMO FUNCIONA EL "SISTEMA DE DOS COMPARTIMENTOS"?
Esta memoria gestiona tu conversación dividiéndola en dos partes:

1. COMPARTIMENTO LITERAL (Buffer - Memoria a corto plazo):
   Guarda tus últimas preguntas y respuestas TAL CUAL ocurrieron, con todo 
   lujo de detalles.

2. COMPARTIMENTO RESUMEN (Moving Summary - Memoria a largo plazo):
   Cuando la conversación crece y supera un límite (max_token_limit), los 
   mensajes MÁS ANTIGUOS se borran del compartimento literal. 
   Pero antes de borrarlos, se envían a una IA interna que escribe un 
   RESUMEN de ellos y lo guarda en este compartimento.

¿POR QUÉ HACEMOS ESTO?
- Si solo usamos Buffer: Gastamos demasiados tokens y dinero en cada pregunta.
- Si solo usamos Resumen: Perdemos los detalles de lo último que acabamos de decir.
- Con esta memoria híbrida: Mantenemos el detalle de lo reciente y la idea 
  general de lo antiguo.

FLUJO DE TRABAJO EN CADA PREGUNTA:
LangChain construye un "Super-Mensaje" para enviar al modelo:
[Resumen acumulado de lo antiguo] + [Últimos mensajes literales] + [Tu pregunta actual]
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage

# 1. CONFIGURACIÓN DEL ENTORNO
load_dotenv()

# 2. INICIALIZACIÓN DEL MODELO
# Necesitamos el LLM para dos cosas: 1. Charlar y 2. Escribir los resúmenes.
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# 3. CREACIÓN DE LA MEMORIA HÍBRIDA
# llm=chat: La IA que resumirá los mensajes antiguos.
# max_token_limit=100: Forzamos un límite bajo para que el resumen se active rápido.
memory = ConversationSummaryBufferMemory(
    llm=chat, 
    max_token_limit=100, 
    memory_key="history", 
    return_messages=True
)

# 4. CREACIÓN DE LA CADENA
# verbose=True es VITAL aquí. Verás en verde cómo LangChain gestiona el resumen.
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

print("--- PRUEBA DE MEMORIA HÍBRIDA (DETALLE + RESUMEN) ---")

# Paso 1: Generamos una conversación larga (muchos tokens)
plan_viaje = (
    "Este fin de semana quiero ir a la playa, a Santiago de la Ribera. "
    "¿Qué me recomiendas hacer? Dime lugares concretos, como restaurantes, "
    "bares, museos o actividades típicas del lugar."
)

print("\nEnviando pregunta detallada...")
response1 = conversation.predict(input=plan_viaje)
print(f"\nRespuesta IA (Detallada):\n{response1}")

# Paso 2: Hacemos una pregunta de seguimiento. 
# Como la respuesta anterior fue larga, superaremos los 100 tokens. 
# LangChain resumirá la 'Interacción 1' para dejar hueco a la 'Interacción 2'.
print("\n--- SEGUNDA PREGUNTA (Activando el resumen automático) ---")
follow_up = "De esos sitios, ¿cuál es el mejor para ir con niños y que tenga buenas vistas?"
response2 = conversation.predict(input=follow_up)
print(f"\nRespuesta IA (Contextualizada):\n{response2}")

# 5. INSPECCIÓN PROFUNDA DE LA MEMORIA
# Aquí es donde verás la magia de los dos compartimentos:
print("\n" + "="*50)
print("ESTADO INTERNO DE LA MEMORIA (CÓMO VE EL MUNDO LA IA)")
print("="*50)

# A) Lo que se ha resumido (Memoria a largo plazo)
# Se guarda en el campo 'moving_summary_buffer'
print(f"\n1. RESUMEN ACUMULADO (Antiguo - Texto resumido):")
print(f"   -> {memory.moving_summary_buffer if memory.moving_summary_buffer else '[Aún no hay resumen, no se ha superado el límite]'}")

# B) Lo que queda literal (Memoria a corto plazo)
# Son los mensajes que aún no han sido "comprimidos" en el resumen.
print("\n2. MENSAJES LITERALES (Recientes - Objetos de mensaje):")
for msg in memory.chat_memory.messages:
    role = "Usuario" if isinstance(msg, HumanMessage) else "IA"
    # Mostramos solo los primeros 80 caracteres para no llenar la pantalla
    content_preview = (msg.content[:80] + '...') if len(msg.content) > 80 else msg.content
    print(f"   [{role}]: {content_preview}")

print("\n💡 NOTA FINAL: Fíjate cómo los mensajes de la primera pregunta han 'desaparecido' "
      "\nde la lista de mensajes literales y ahora forman parte del texto del resumen.")