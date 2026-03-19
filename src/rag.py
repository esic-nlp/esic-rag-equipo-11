import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# Inicializamos el modelo de lenguaje de forma global para mejorar la eficiencia
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def crear_indice(df):
    """
    Transforma el texto de búsqueda en vectores y crea el índice FAISS.
    Recibe: DataFrame con la columna 'texto_busqueda'.
    Retorna: El índice FAISS listo para consultas.
    """
    print("[RAG] Generando embeddings e índice FAISS...")
    embeddings = embedder.encode(df["texto_busqueda"].tolist(), show_progress_bar=False)

    # Creamos un índice de tipo L2 (distancia euclidiana)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype("float32"))

    return index


def buscar_y_responder(consulta, df, index):
    """
    Busca los productos más relevantes y aplica el ranking personalizado.
    Recibe: consulta (str), dataframe procesado y el índice FAISS.
    """

    consulta_lower = consulta.lower()

    # 🧠 Detectar atributos pedidos (multi-intención)
    atributos = {
        "proteinas": ["proteina", "proteinas"],
        "grasas": ["grasa", "grasas"],
        "calorias": ["caloria", "calorias"],
        "sal": ["sal", "sodio"],
        "precio": ["precio", "barato", "caro"],
        "salud": ["salud", "healthy", "sano"]
    }

    atributos_detectados = []
    for key, keywords in atributos.items():
        if any(k in consulta_lower for k in keywords):
            atributos_detectados.append(key)

    # 1. Búsqueda Vectorial
    vec_query = embedder.encode([consulta]).astype("float32")
    dist, indices = index.search(vec_query, 20)  # 🔥 aumentamos candidatos

    candidatos = df.iloc[indices[0]].copy()

    # 2. Re-ranking (Normalización local de distancias)
    max_dist = dist[0].max() if dist[0].max() > 0 else 1
    candidatos["norm_dist"] = 1 - (dist[0] / max_dist)

    # Ranking base
    candidatos["rank_final"] = (
        candidatos["norm_dist"] * 0.6
        + candidatos["norm_nutri"] * 0.2
        + candidatos["norm_precio"] * 0.2
    )

    # 🔥 Ajuste ranking si piden atributos específicos
    if "proteinas" in atributos_detectados:
        candidatos = candidatos.sort_values("proteinas", ascending=False)
    if "grasas" in atributos_detectados:
        candidatos = candidatos.sort_values("grasas", ascending=True)
    if "precio" in atributos_detectados:
        candidatos = candidatos.sort_values("precio", ascending=True)

    # 🔥 TOP 8 productos
    mejores = candidatos.sort_values("rank_final", ascending=False).head(8)

    # 3. Formateo dinámico de la respuesta
    contexto = ""

    for _, r in mejores.iterrows():

        # MODO GENERAL (no atributos detectados)
        if not atributos_detectados:
            contexto += (
                f"- {r['titulo']} | Precio: {r['precio']}€ | "
                f"Proteínas: {r['proteinas']}g | "
                f"Grasas: {r.get('grasas', 'N/A')}g | "
                f"Sal: {r.get('sal', 'N/A')}g | "
                f"Salud: {int(r['score_nutricional'])}/100\n"
            )
        else:
            linea = f"- {r['titulo']}"
            if "proteinas" in atributos_detectados:
                linea += f" | Proteínas: {r['proteinas']}g"
            if "grasas" in atributos_detectados:
                linea += f" | Grasas: {r.get('grasas', 'N/A')}g"
            if "calorias" in atributos_detectados:
                linea += f" | Calorías: {r.get('calorias', 'N/A')} kcal"
            if "sal" in atributos_detectados:
                linea += f" | Sal: {r.get('sal', 'N/A')}g"
            if "precio" in atributos_detectados:
                linea += f" | Precio: {r['precio']}€"
            if "salud" in atributos_detectados:
                linea += f" | Salud: {int(r['score_nutricional'])}/100"

            contexto += linea + "\n"

    return f"**Asistente Nutricional:** Para '{consulta}', he encontrado estas opciones:\n\n{contexto}"


def consultar(df):
    """
    Función principal para ejecutar el RAG.
    """
    id = crear_indice(df)
    while True:
        consulta = input("Introduce tu consulta (o 'salir' para terminar): ")
        if consulta.lower() == "salir":
            print("¡Hasta luego!")
            break
        respuesta = buscar_y_responder(consulta, df, id)
        print(respuesta)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df=pd.read_csv("data/clean/products_clean.csv")
    df_clean = consultar(df)

