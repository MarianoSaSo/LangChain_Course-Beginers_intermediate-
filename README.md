# 🚀 Proyectos Finales: Curso de LangChain (Agentes Inteligentes)

Este repositorio contiene la culminación del curso de LangChain, donde se aplican todos los conceptos aprendidos (RAG, Memoria, Herramientas y Razonamiento) en dos casos de uso de nivel profesional.

---

## 🛠️ Proyecto Final 1: Agente Híbrido con RAG (Pinecone) y Wikipedia
**Ubicación:** `Leccion 22 (Proyecto Agente 1)/`

Un agente capaz de actuar como el "cerebro" de una organización, decidiendo inteligentemente entre diferentes fuentes de información.

### Características Clave:
*   **RAG con Pinecone:** Conecta con una base de datos vectorial que contiene el PDF de "Historia de España".
*   **Razonamiento Crítico:** El agente evalúa si la respuesta está en los documentos internos o si debe buscar en Wikipedia.
*   **Memoria de Conversación:** Utiliza `ConversationBufferMemory` para mantener el hilo de preguntas complejas y recordatorios históricos.
*   **Personalización de Herramientas:** Definición de una `Custom Tool` con descripciones de "autoridad" para priorizar fuentes oficiales.

---

## 📊 Proyecto Final 2: Agente SQL - Análisis de Datos en Lenguaje Natural
**Ubicación:** `Leccion 23 (Proyecto Agente 2)/`

La eliminación de la barrera técnica entre el usuario y las bases de datos relacionales (MySQL).

### Características Clave:
*   **Traducción Text-to-SQL:** El agente traduce peticiones en español (ej: "Dime el promedio de esperanza de vida") a queries SQL complejas de forma autónoma.
*   **Schema Awareness:** Capacidad del agente para explorar la estructura de las tablas (`world` database) antes de ejecutar consultas.
*   **Seguridad y Mejores Prácticas:** Implementación de carga de contraseñas mediante archivos externos (`password_sql.txt`) para evitar exposición de credenciales.
*   **Integración MySQL:** Conexión robusta mediante `SQLAlchemy` y `mysql-connector-python`.

---

## 📝 Instrucciones de Instalación
Para ejecutar estos proyectos, asegúrate de tener instaladas las dependencias necesarias:
```bash
pip install mysql-connector-python sqlalchemy pinecone-client langchain-openai
```

## 🔐 Archivos de Configuración
Asegúrate de tener en la raíz del proyecto:
1. `.env`: Con tus API Keys (OpenAI y Pinecone).
2. `password_sql.txt`: Con la contraseña de tu base de datos local (ej: `123456`).

---
¡Gracias por completar este curso! Ahora tienes las herramientas para construir sistemas de IA autónomos y potentes.
