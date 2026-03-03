"""
LECCIÓN 10: Gestión de Memoria - El objeto ChatMessageHistory
------------------------------------------------------------
En esta lección exploraremos la unidad más básica de memoria en LangChain.
A diferencia de las "Cadenas con Memoria" (que veremos más adelante), 
ChatMessageHistory es un contenedor manual que nos permite gestionar el 
historial de mensajes paso a paso.

¿QUÉ ES CHATMESSAGEHISTORY?
Es una clase sencilla que almacena una lista de objetos de mensaje 
(HumanMessage, AIMessage, etc.). Su trabajo es simplemente "recordar" 
lo que se ha dicho, pero no lo envía automáticamente al modelo.

REQUISITOS:
- Tener instalado 'langchain_community'.
- Importar 'ChatMessageHistory' desde 'langchain_community.chat_message_histories'.

VENTAJAS:
1. Control Total: Tú decides exactamente qué mensaje se guarda y cuándo.
2. Transparencia: Es ideal para aprender cómo funciona el flujo de mensajes.
3. Versatilidad: Puedes usarlo en scripts simples sin necesidad de crear 
   estructuras complejas de cadenas (chains).

INCONVENIENTES:
1. Gestión Manual: Si olvidas llamar a 'add_user_message' o 'add_ai_message', 
   la conversación se pierde.
2. Sin gestión de ventana: Si la conversación es muy larga, el historial 
   crecerá indefinidamente hasta sobrepasar el límite de tokens del modelo.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. CONFIGURACIÓN E INICIALIZACIÓN
load_dotenv()
# Creamos el modelo de lenguaje (LLM)
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2. CREACIÓN DEL OBJETO DE HISTORIAL
# Este objeto es como una "libreta" vacía donde anotaremos la conversación.
history = ChatMessageHistory()

# 3. INTERACCIÓN CON EL USUARIO
consulta = "Hola, ¿cuál es la capital de España?"

# 4. GUARDADO MANUAL DEL MENSAJE DEL USUARIO
# Importante: ChatMessageHistory no "escucha" la conversación, 
# tenemos que decirle qué guardar.
history.add_user_message(consulta)

# 5. LLAMADA AL MODELO
# Enviamos la consulta actual al modelo.
# Nota: En esta lección enviamos solo el mensaje actual, 
# en futuras lecciones enviaremos TODO el historial.
resultado = chat.invoke([HumanMessage(content=consulta)])

# 6. GUARDADO MANUAL DE LA RESPUESTA DE LA IA
# Guardamos la respuesta para que nuestro historial esté completo.
history.add_ai_message(resultado.content)

# 7. VISUALIZACIÓN DEL ESTADO DE LA MEMORIA
# Imprimimos la lista de mensajes almacenados.
print("--- HISTORIAL DE LA CONVERSACIÓN ---")
for msg in history.messages:
    # Identificamos el tipo de mensaje para una lectura más clara
    role = "Usuario" if isinstance(msg, HumanMessage) else "IA"
    print(f"{role}: {msg.content}")
