# 🛒 NutriSearch — Asistente Nutricional de Supermercados

Sistema RAG (Retrieval Augmented Generation) que permite consultar productos del supermercado Ahorramas por sus propiedades nutricionales, precio y categoría, a través de una interfaz web interactiva.

---

## 📋 Requisitos previos

- Python 3.10 o superior
- Git
- Tesseract OCR (solo si se quiere volver a ejecutar el scraper)

---

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/esic-nlp/esic-rag-equipo-11.git
cd esic-rag-equipo-11
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el pipeline completo

```bash
python main.py
```

Esto ejecutará automáticamente:
1. **Preprocessing** → limpia y normaliza `src/data/raw/ahorramas_products.json`
2. **RAG** → genera embeddings, crea índice FAISS y lanza la interfaz web

La interfaz se abrirá automáticamente en **http://localhost:5000**

---

## 📁 Estructura del proyecto

```
esic-rag-equipo-11/
├── main.py                          # Punto de entrada del pipeline
├── requirements.txt                 # Dependencias
├── README.md
├── data/
│   └── ejemplo.json                 # Ejemplo de estructura de datos
└── src/
    ├── acquisition.py               # Scraper de Ahorramas
    ├── preprocessing.py             # Limpieza y normalización de datos
    ├── rag.py                       # Sistema RAG + interfaz web (Flask)
    ├── ACQUISITION.ipynb            # Notebook del scraper
    └── data/
        ├── raw/
        │   └── ahorramas_products.json     # Datos crudos scrapeados
        └── clean/
            ├── products_clean.csv          # Datos procesados (CSV)
            └── products_clean.json         # Datos procesados (JSON)
```

---

## 🔧 Componentes principales

### acquisition.py
Scraper para **www.ahorramas.com**. Extrae productos con título, precio e información nutricional mediante OCR sobre las imágenes de tabla nutricional.

Para volver a ejecutar el scraper (no necesario, los datos ya están incluidos):
```bash
# Instalar Tesseract OCR primero:
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux:   sudo apt-get install tesseract-ocr tesseract-ocr-spa

pip install pytesseract Pillow
cd src
python acquisition.py
```

### preprocessing.py
Limpia y transforma los datos crudos. Genera las columnas necesarias para el RAG:

| Columna | Descripción |
|---|---|
| `titulo` | Nombre del producto (normalizado) |
| `precio` | Precio en euros |
| `proteinas` | Proteínas por 100g |
| `carbohidratos` | Carbohidratos por 100g |
| `grasas` | Grasas por 100g |
| `fibra` | Fibra alimentaria por 100g |
| `calories` | Valor energético en kcal |
| `texto_busqueda` | Texto para embeddings semánticos |
| `norm_precio` | Precio normalizado 0-1 (inverso) |
| `norm_nutri` | Score proteínas 0-100 |
| `score_nutricional` | Score nutricional agregado 0-100 |

### rag.py
Sistema RAG con interfaz web (Flask). Implementa:
- Embeddings semánticos con `sentence-transformers`
- Búsqueda vectorial con FAISS
- Re-ranking: **60% semántica + 20% nutrición + 20% precio**
- Interfaz web con filtros por cocina, precio y proteínas
- Filtros por tipo de cocina: 🇮🇹 Italiana, 🇲🇽 Mexicana, 🇺🇸 Americana, 🇯🇵 Japonesa, 🇬🇷 Mediterránea, 🇮🇳 India, 🇨🇳 China, 🥗 Saludable

---

## 📦 Dependencias principales

```
faiss-cpu
numpy
sentence-transformers
pandas
flask
requests
beautifulsoup4
pytesseract
Pillow
```

---

## 💻 Uso de la interfaz web

Una vez ejecutado `python main.py`, se abre automáticamente el navegador en `http://localhost:5000`.

**Funcionalidades:**
- 🔍 **Buscador semántico** — escribe en lenguaje natural: *"algo rico en proteínas para después del gym"*
- 🌍 **Filtro por cocina** — selecciona el tipo de cocina para filtrar productos relevantes
- 💰 **Filtro por precio máximo** — slider de 0 a 50€
- 💪 **Filtro por proteínas mínimas** — filtra por contenido proteico
- 🥗 **Solo con nutrición** — muestra únicamente productos con datos nutricionales completos
- 📊 **Score nutricional** — cada producto muestra su puntuación nutricional visual

---

## 👥 Equipo

Equipo 11 — ESIC University  
Asignatura: Procesamiento del Lenguaje Natural

