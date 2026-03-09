# 🦜 Curso Completo de LangChain: De Principiante a Experto (Agentes Inteligentes)

¡Bienvenido al curso de LangChain! Este repositorio contiene todo el material necesario para aprender a construir aplicaciones de IA de última generación, utilizando modelos de lenguaje (LLMs), bases de datos vectoriales, memoria y agentes autónomos.

Este curso está diseñado para llevarte paso a paso, desde los conceptos más básicos hasta la implementación de proyectos profesionales complejos.

---

## 📚 Estructura del Curso

El curso se divide en lecciones prácticas que cubren todo el ecosistema de LangChain:

### 🛠️ Fundamentos y Estructura
- **Lección 1**: Modelos de entrada y salida (System vs Human Messages).
- **Lección 2**: Plantillas de Prompts (`ChatPromptTemplate`).
- **Lección 3**: Output Parsers (Procesamiento de respuestas).

### 📄 Gestión de Documentos y RAG
- **Lección 4 y 5**: Cargadores de datos y documentos externos.
- **Lección 6**: Transformación de documentos (Chunking/Text Splitting).
- **Lección 7**: Creación de Embeddings (Representación vectorial).
- **Lección 8**: Almacenamiento en Bases de Datos Vectoriales (Pinecone/VectorStores).
- **Lección 9**: Integración completa: Almacenamiento + Chat + Memoria.

### 🧠 Memoria de Conversación
- **Lección 10**: `ChatMessageHistory`.
- **Lección 11**: `ConversationBufferMemory`.
- **Lección 12**: `ConversationBufferWindowMemory` (Memoria de ventana corta).
- **Lección 14**: `ConversationSummaryBufferMemory` (Resumen inteligente de chats).

### 🤖 Agentes y LCEL (Modern LangChain)
- **Lección 15 a 18**: Introducción a Agentes (ReAct Pattern) y uso de herramientas.
- **Lección 19**: **LCEL (LangChain Expression Language)** - El estándar moderno para encadenar componentes.
- **Lección 20**: Agente con Memoria y despliegue con API (FastAPI).
- **Lección 21**: Agente con Memoria Persistente (Pickle) y API.

---

## 🏆 Proyectos Finales

### 1. Agente Híbrido: RAG + Wikipedia (Lección 22)
Un agente capaz de actuar como el "cerebro" de una organización. Decide inteligentemente si buscar en documentos internos (Pinecone) o en fuentes externas (Wikipedia).
- **Tecnologías**: Pinecone, Wikipedia API, Custom Tooling, Memory.

### 2. Agente SQL: Análisis de Datos (Lección 23)
Elimina la barrera entre el lenguaje humano y las bases de datos relacionales. El agente traduce preguntas en español a consultas SQL reales.
- **Tecnologías**: MySQL, SQLAlchemy, `create_sql_agent`, Schema Awareness.

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/MarianoSaSo/LangChain_Course-Beginers_intermediate-
cd LangChain_Curso
```

### 2. Crear entorno virtual
```ps1
# Crear entorno
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configuración de API Keys
Crea un archivo `.env` en la raíz del proyecto con la siguiente estructura:
```env
OPENAI_API_KEY="tu_clave_de_openai"
PINECONE_API_KEY="tu_clave_de_pinecone"
LANGSMITH_API_KEY="tu_clave_de_langsmith" # Opcional
```

### 5. Configuración para el Agente SQL
Para la Lección 23, asegúrate de tener:
- Una instancia de MySQL local instalada.
- El archivo `password_sql.txt` en la raíz con la contraseña de tu base de datos `root`.

---

## 🔐 Seguridad y Mejores Prácticas

**IMPORTANTE**: Este repositorio está configurado para ignorar archivos sensibles. Nunca compartas tus archivos `.env`, `password_sql.txt` o la carpeta `venv/`. Asegúrate de que el archivo `.gitignore` sea respetado antes de subir cambios a tu propio repositorio.

---

## ✨ Créditos
Curso creado por **Mariano SaSo**. ¡Espero que disfrutes aprendiendo LangChain tanto como yo disfruté creando este material!
