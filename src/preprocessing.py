import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURACION GENERAL
# ============================================================================
# En esta sección se definen las rutas de entrada y salida del proceso.
# El script leerá el archivo JSON generado en la fase de adquisición y
# guardará el resultado ya procesado en la carpeta data/clean.
#
# INPUT_FILE:
#   Archivo crudo generado por el scraper.
#
# OUTPUT_FILE:
#   Archivo final limpio y transformado, listo para utilizarse en el sistema
#   de recuperación de información o en cualquier pipeline posterior.
# ============================================================================

INPUT_FILE = os.path.join("data", "raw", "ahorramas_products.json")
OUTPUT_DIR = os.path.join("data", "clean")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ahorramas_products_clean.csv")


# ============================================================================
# COLUMNAS FINALES
# ============================================================================
# Estas son las columnas que tendrá el DataFrame final.
# Se incluyen los datos nutricionales principales, el texto de búsqueda para
# embeddings, las variables normalizadas y el score nutricional agregado.
#
# También se conserva la categoría principal para que el dataset sea más fácil
# de analizar, depurar y reutilizar después.
# ============================================================================

OUTPUT_COLUMNS = [
    "titulo",
    "precio",
    "proteinas",
    "carbohidratos",
    "grasas",
    "fibra",
    "calories",
    "texto_busqueda",
    "norm_precio",
    "norm_nutri",
    "score_nutricional",
    "categoria_principal",
]


# ============================================================================
# CARGA DEL JSON
# ============================================================================
# Esta función carga el archivo JSON crudo generado por el scraper.
# Se comprueba que el contenido sea una lista de productos, ya que ese es
# el formato esperado para poder construir el DataFrame posterior.
# ============================================================================

def load_raw_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El archivo JSON debe contener una lista de productos.")

    return data


# ============================================================================
# NORMALIZACION DE TEXTO
# ============================================================================
# Esta función se utiliza para limpiar y uniformar cualquier campo textual.
# Convierte el texto a minúsculas, elimina acentos y normaliza espacios.
#
# Esto es especialmente útil para:
# - evitar inconsistencias entre registros
# - mejorar la calidad del texto de búsqueda
# - facilitar comparaciones y agrupaciones
# ============================================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================================
# EXTRACCION DE NUMEROS
# ============================================================================
# Muchos valores procedentes del scraper llegan como texto, por ejemplo:
# "12 g", "3,5 g", "1,99 €" o "250 kcal".
#
# Esta función extrae la parte numérica y la convierte a float para poder
# trabajar después con operaciones matemáticas y estadísticas.
# Si no se puede extraer un número válido, devuelve NaN.
# ============================================================================

def extract_number(value: Any) -> Optional[float]:
    if value is None:
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower().replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+", text)

    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan

    return np.nan


# ============================================================================
# EXTRACCION DE NUTRIENTES
# ============================================================================
# Los valores nutricionales vienen almacenados dentro de un diccionario.
# Esta función permite buscar un nutriente concreto probando varias posibles
# claves, ya que en algunos casos puede haber pequeñas variaciones de nombre.
#
# Si encuentra el nutriente, devuelve su valor numérico.
# Si no lo encuentra, devuelve NaN.
# ============================================================================

def get_nutrient(nutrients: Dict[str, Any], possible_keys: List[str]) -> Optional[float]:
    if not isinstance(nutrients, dict):
        return np.nan

    for key in possible_keys:
        if key in nutrients:
            return extract_number(nutrients.get(key))

    return np.nan


# ============================================================================
# EXTRACCION DE MARCA
# ============================================================================
# Como el JSON original no siempre trae una columna de marca separada,
# se toma una aproximación sencilla: usar la primera palabra del título
# como posible marca.
#
# No es perfecto, pero resulta útil para enriquecer el campo
# texto_busqueda cuando no existe una marca explícita.
# ============================================================================

def extract_brand_from_title(title: str) -> str:
    if not title:
        return ""

    words = str(title).strip().split()
    if not words:
        return ""

    return words[0].lower()


# ============================================================================
# MAPEO DE CATEGORIA PRINCIPAL
# ============================================================================
# Las categorías originales del scraper son muy específicas, por ejemplo:
# "pasta_corta", "arroz_redondo", "quesos_semicurados".
#
# Para poder imputar valores faltantes de forma más coherente, se agrupan
# en categorías principales más generales, como:
# pasta, arroz, carne, bebidas, congelados, quesos, etc.
#
# Esta agrupación permite calcular medianas más realistas entre productos
# similares y evita mezclar alimentos de naturaleza muy distinta.
# ============================================================================

def map_main_category(raw_category: str) -> str:
    raw_category = normalize_text(raw_category)

    if "pasta" in raw_category:
        return "pasta"
    if "arroz" in raw_category:
        return "arroz"
    if "legumbre" in raw_category or "lenteja" in raw_category:
        return "legumbres"
    if "aceite" in raw_category:
        return "aceites"
    if "atun" in raw_category or "pescado" in raw_category or "marisco" in raw_category:
        return "pescado"
    if "queso" in raw_category:
        return "quesos"
    if "huevo" in raw_category:
        return "huevos"
    if "refresco" in raw_category or "zumo" in raw_category or "bebida" in raw_category:
        return "bebidas"
    if "cerveza" in raw_category:
        return "bebidas"
    if "helado" in raw_category or "pizza" in raw_category or "congelado" in raw_category:
        return "congelados"
    if "carniceria" in raw_category or "ternera" in raw_category or "pollo" in raw_category:
        return "carne"
    if "charcuteria" in raw_category or "embutido" in raw_category or "fiambre" in raw_category:
        return "charcuteria"
    if "galleta" in raw_category or "cereal" in raw_category:
        return "desayuno_snacks"
    if "cafe" in raw_category:
        return "cafe_infusiones"
    if "alimentacion_ofertas" in raw_category:
        return "alimentacion_general"
    if "frescos_ofertas" in raw_category:
        return "frescos"
    if "bebidas_ofertas" in raw_category:
        return "bebidas"
    if "congelados_ofertas" in raw_category:
        return "congelados"

    return "otros"


# ============================================================================
# TEXTO DE BUSQUEDA
# ============================================================================
# Esta función construye la columna texto_busqueda, que es una concatenación
# de la información más relevante de cada producto:
# - título
# - marca
# - descripción
# - categoría principal
#
# El objetivo es disponer de un texto compacto pero rico semánticamente,
# que pueda utilizarse después para generar embeddings en el sistema RAG.
# ============================================================================

def build_texto_busqueda(row: pd.Series) -> str:
    parts = [
        normalize_text(row.get("titulo", "")),
        normalize_text(row.get("marca", "")),
        normalize_text(row.get("descripcion", "")),
        normalize_text(row.get("categoria_principal", "")),
    ]

    text = " ".join(part for part in parts if part)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# NORMALIZACION INVERSA DEL PRECIO
# ============================================================================
# Esta función aplica una normalización min-max inversa sobre el precio.
#
# El resultado queda entre 0 y 1, pero con una lógica invertida:
# - los productos más baratos obtienen valores más altos
# - los productos más caros obtienen valores más bajos
#
# Esto es útil cuando se quiere priorizar productos económicos dentro de un
# sistema de ranking o recomendación.
# ============================================================================

def normalize_inverse_minmax(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([np.nan] * len(series), index=series.index)

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return (max_val - series) / (max_val - min_val)


# ============================================================================
# NORMALIZACION DEL CONTENIDO PROTEICO
# ============================================================================
# Esta función transforma la variable proteínas a una escala entre 0 y 100.
#
# Se utiliza la proteína como referencia principal de calidad nutricional
# porque es un macronutriente especialmente relevante en el contexto del
# proyecto y resulta útil para comparar alimentos de forma sencilla.
# ============================================================================

def normalize_protein_to_100(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([np.nan] * len(series), index=series.index)

    if max_val == min_val:
        return pd.Series([100.0] * len(series), index=series.index)

    return ((series - min_val) / (max_val - min_val)) * 100.0


# ============================================================================
# SCORE NUTRICIONAL
# ============================================================================
# Esta función calcula una puntuación nutricional agregada basada en varios
# macronutrientes y calorías.
#
# La lógica utilizada es:
# - más proteínas aumenta la puntuación
# - más fibra aumenta la puntuación
# - más grasas penaliza la puntuación
# - más carbohidratos penaliza ligeramente la puntuación
# - más calorías también penaliza
#
# Finalmente, el score se reescala entre 0 y 100 para que resulte más fácil
# de interpretar y comparar entre productos.
# ============================================================================

def compute_score_nutricional(df: pd.DataFrame) -> pd.Series:
    proteinas = df["proteinas"].fillna(0)
    fibra = df["fibra"].fillna(0)
    grasas = df["grasas"].fillna(0)
    carbohidratos = df["carbohidratos"].fillna(0)
    calories = df["calories"].fillna(0)

    score = (
        proteinas * 4.0
        + fibra * 3.0
        - grasas * 1.5
        - carbohidratos * 0.3
        - (calories / 50.0)
    )

    score = score.clip(lower=0)

    min_val = score.min()
    max_val = score.max()

    if max_val == min_val:
        return pd.Series([100.0] * len(df), index=df.index)

    score_scaled = ((score - min_val) / (max_val - min_val)) * 100.0
    return score_scaled.round(2)


# ============================================================================
# IMPUTACION POR MEDIANA SEGUN CATEGORIA
# ============================================================================
# Esta función rellena los valores faltantes de determinadas columnas
# numéricas utilizando la mediana de su categoría principal.
#
# Por ejemplo:
# - si a una pasta le faltan proteínas, se usa la mediana de proteínas
#   del resto de productos de la categoría pasta
# - si no hay suficientes datos en esa categoría, se usa la mediana global
#
# Este enfoque es más coherente que imputar con una media o mediana global
# desde el principio, ya que respeta mejor las diferencias naturales entre
# tipos de alimentos.
# ============================================================================

def impute_by_category_median(df: pd.DataFrame, cols: List[str], category_col: str) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        category_medians = df.groupby(category_col)[col].median()
        global_median = df[col].median()

        df[col] = df.apply(
            lambda row: (
                row[col]
                if pd.notna(row[col])
                else category_medians.get(row[category_col], np.nan)
            ),
            axis=1,
        )

        if pd.notna(global_median):
            df[col] = df[col].fillna(global_median)

    return df


# ============================================================================
# PREPROCESAMIENTO PRINCIPAL
# ============================================================================
# Esta es la función central del script. Se encarga de:
#
# 1. transformar el JSON crudo en un DataFrame
# 2. extraer las variables más relevantes
# 3. limpiar textos y valores numéricos
# 4. eliminar duplicados
# 5. imputar valores faltantes por categoría
# 6. crear variables enriquecidas y normalizadas
# 7. devolver el DataFrame final listo para guardar
#
# Es la parte principal de la fase de preprocessing del proyecto.
# ============================================================================

def preprocess_products(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for product in raw_data:
        nutrients = product.get("valores_nutricionales_100_g", {})
        categorias = product.get("categorias", [])

        categoria_raw = categorias[0] if isinstance(categorias, list) and categorias else ""

        titulo = product.get("titulo", "")
        descripcion = product.get("descripcion", "")
        marca = extract_brand_from_title(titulo)

        row = {
            "titulo": titulo,
            "descripcion": descripcion,
            "marca": marca,
            "precio": extract_number(product.get("precio_total")),
            "proteinas": get_nutrient(nutrients, ["Proteinas"]),
            "carbohidratos": get_nutrient(nutrients, ["Hidratos de carbono"]),
            "grasas": get_nutrient(nutrients, ["Grasas"]),
            "fibra": get_nutrient(nutrients, ["Fibra alimentaria"]),
            "calories": get_nutrient(nutrients, ["Valor energetico"]),
            "categoria": categoria_raw,
            "categoria_principal": map_main_category(categoria_raw),
            "url": product.get("url", ""),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # Limpieza y normalización de textos
    # ------------------------------------------------------------------------
    # Se aplica la función de limpieza textual a las columnas de texto para
    # evitar inconsistencias y dejar los valores en un formato homogéneo.
    # ------------------------------------------------------------------------
    df["titulo"] = df["titulo"].apply(normalize_text)
    df["descripcion"] = df["descripcion"].apply(normalize_text)
    df["marca"] = df["marca"].apply(normalize_text)
    df["categoria"] = df["categoria"].apply(normalize_text)
    df["categoria_principal"] = df["categoria_principal"].apply(normalize_text)

    # ------------------------------------------------------------------------
    # Conversión a numérico
    # ------------------------------------------------------------------------
    # Las columnas nutricionales y de precio se convierten a formato numérico
    # para permitir operaciones estadísticas y cálculos posteriores.
    # ------------------------------------------------------------------------
    numeric_cols = ["precio", "proteinas", "carbohidratos", "grasas", "fibra", "calories"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------------
    # Eliminación de duplicados
    # ------------------------------------------------------------------------
    # Se eliminan filas repetidas usando como referencia el título, el precio
    # y la URL, ya que esa combinación identifica razonablemente a un producto.
    # ------------------------------------------------------------------------
    df = df.drop_duplicates(subset=["titulo", "precio", "url"]).copy()

    # ------------------------------------------------------------------------
    # Eliminación de filas con faltantes críticos
    # ------------------------------------------------------------------------
    # Solo se eliminan productos que no tienen título o precio, porque son
    # campos esenciales. Los nutrientes faltantes no se eliminan todavía,
    # ya que después serán imputados mediante medianas por categoría.
    # ------------------------------------------------------------------------
    df = df.dropna(subset=["titulo", "precio"]).copy()
    df = df[df["titulo"] != ""].copy()
    df = df[df["precio"] > 0].copy()

    # ------------------------------------------------------------------------
    # Imputación de nutrientes faltantes por categoría principal
    # ------------------------------------------------------------------------
    # En este paso se rellenan los valores faltantes de proteínas,
    # carbohidratos, grasas, fibra y calorías utilizando la mediana de
    # productos similares dentro de la misma categoría principal.
    # ------------------------------------------------------------------------
    df = impute_by_category_median(
        df,
        cols=["proteinas", "carbohidratos", "grasas", "fibra", "calories"],
        category_col="categoria_principal",
    )

    # ------------------------------------------------------------------------
    # Construcción del texto de búsqueda
    # ------------------------------------------------------------------------
    # Se crea una columna con toda la información textual relevante del
    # producto, que posteriormente podrá usarse para embeddings y búsqueda
    # semántica dentro del sistema RAG.
    # ------------------------------------------------------------------------
    df["texto_busqueda"] = df.apply(build_texto_busqueda, axis=1)

    # ------------------------------------------------------------------------
    # Normalizaciones y score nutricional
    # ------------------------------------------------------------------------
    # Se calculan tres variables nuevas:
    # - norm_precio: prioriza los productos más baratos
    # - norm_nutri: escala de proteínas de 0 a 100
    # - score_nutricional: indicador agregado basado en macronutrientes
    # ------------------------------------------------------------------------
    df["norm_precio"] = normalize_inverse_minmax(df["precio"]).round(4)
    df["norm_nutri"] = normalize_protein_to_100(df["proteinas"]).round(2)
    df["score_nutricional"] = compute_score_nutricional(df)

    # ------------------------------------------------------------------------
    # Selección de columnas finales
    # ------------------------------------------------------------------------
    # Se conservan únicamente las variables necesarias para el sistema final,
    # reduciendo ruido y dejando un dataset más limpio y manejable.
    # ------------------------------------------------------------------------
    df = df[OUTPUT_COLUMNS].copy()

    return df


# ============================================================================
# GUARDADO DEL DATAFRAME
# ============================================================================
# Esta función crea la carpeta de salida si no existe y guarda el resultado
# final en formato CSV.
#
# El CSV generado constituye la versión limpia y tratada del dataset,
# preparada para alimentar el sistema RAG o cualquier otro módulo analítico.
# ============================================================================

def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================
# La función main coordina todo el flujo:
# - carga los datos crudos
# - ejecuta el preprocessing
# - guarda el resultado limpio
# - muestra por pantalla un pequeño resumen final
#
# Esto permite ejecutar todo el pipeline de transformación con un único
# comando desde terminal.
# ============================================================================

def main() -> None:
    raw_data = load_raw_json(INPUT_FILE)
    df_clean = preprocess_products(raw_data)
    save_dataframe(df_clean, OUTPUT_FILE)

    print("Preprocessing completado correctamente.")
    print(f"Filas finales: {len(df_clean)}")
    print(f"Archivo guardado en: {OUTPUT_FILE}")
    print("\nColumnas finales:")
    print(list(df_clean.columns))
    print("\nPrimeras filas:")
    print(df_clean.head())


if __name__ == "__main__":
    main()