"""
preprocessing.py
================
Limpia, transforma y enriquece los datos de ahorramas_products.json
para su uso en el pipeline RAG.

Entrada:  src/data/raw/ahorramas_products.json
Salida:   src/data/clean/products_clean.csv
          src/data/clean/products_clean.json

Uso:
    python preprocessing.py
    o desde otro módulo:
        from src.preprocessing import procesar_datos
        df = procesar_datos()
"""

import json
import os
import re
import pandas as pd
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Rutas — relativas a la ubicación de este script (src/)
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "data", "raw", "ahorramas_products.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "clean")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "products_clean.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "products_clean.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value) -> Optional[float]:
    """'6,7 g' | '537 kcal' | 30.4 | None  →  float o None"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if not (isinstance(value, float) and np.isnan(value)) else None
    m = re.search(r"(\d+[.,]\d+|\d+)", str(value).replace(",", "."))
    return float(m.group(1)) if m else None


def _clean_text(text) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáéíóúüñ,.()/%-]", "", text)
    return text

# ---------------------------------------------------------------------------
# Mapa de claves del JSON → columna destino
# ---------------------------------------------------------------------------
NUTRIENT_MAP = {
    "calorias":      ["Valor energetico", "Energía", "Calorías"],
    "calorias_kj":   ["Valor energetico en KJ"],
    "grasas":        ["Grasas", "Grasa total"],
    "saturadas":     ["Saturadas", "Ácidos grasos saturados"],
    "carbohidratos": ["Hidratos de carbono", "Carbohidratos"],
    "azucares":      ["Azucares", "Azúcares"],
    "fibra":         ["Fibra alimentaria", "Fibra dietética", "Fibra"],
    "proteinas":     ["Proteinas", "Proteínas"],
    "sal":           ["Sal", "Sodio"],
}

# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_raw(path: str = INPUT_FILE) -> pd.DataFrame:
    print(f"📂 Cargando: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"   Registros cargados: {len(df)}")
    return df

# ---------------------------------------------------------------------------
# Extracción de nutrientes
# ---------------------------------------------------------------------------

def extract_nutrients(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte valores_nutricionales_100_g (dict) en columnas float."""
    nutri_col = "valores_nutricionales_100_g"
    if nutri_col not in df.columns:
        df[nutri_col] = [{} for _ in range(len(df))]

    def _get(d: dict, keys: list):
        if not isinstance(d, dict):
            return None
        for k in keys:
            if k in d:
                return _to_float(d[k])
        return None

    for col_name, keys in NUTRIENT_MAP.items():
        df[col_name] = df[nutri_col].apply(lambda d: _get(d, keys))

    return df

# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 LIMPIEZA")
    n0 = len(df)

    df = extract_nutrients(df)

    df = df.rename(columns={
        "precio_total":        "precio",
        "precio_por_cantidad": "precio_por_kg",
    })

    for col in ["precio", "precio_por_kg", "calorias", "calorias_kj",
                "grasas", "saturadas", "carbohidratos", "azucares",
                "fibra", "proteinas", "sal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["titulo", "precio"])
    print(f"   Eliminadas sin titulo/precio: {n0 - len(df)}")

    n1 = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["titulo"], keep="first")
    print(f"   Duplicados eliminados: {n1 - len(df)}")

    df["titulo"]      = df["titulo"].apply(_clean_text)
    df["descripcion"] = df.get("descripcion", pd.Series("", index=df.index)).fillna("").apply(_clean_text)
    df["categoria"]   = df["categorias"].apply(
        lambda x: x[0] if isinstance(x, list) and x else "otros"
    )

    print(f"   Registros tras limpieza: {len(df)}")
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📐 NORMALIZACIÓN")

    df["texto_busqueda"] = (
        df["titulo"].fillna("") + " " +
        df["categoria"].fillna("") + " " +
        df["descripcion"].fillna("")
    ).str.strip().apply(_clean_text)

    p = df["precio"]
    p_min, p_max = p.min(), p.max()
    df["norm_precio"] = (1 - (p - p_min) / (p_max - p_min)).round(4) if p_max > p_min else 0.5

    prot        = df["proteinas"].fillna(0)
    precio_safe = df["precio"].replace(0, np.nan)
    prot_per_eur = prot / precio_safe
    mn, mx = prot_per_eur.min(), prot_per_eur.max()
    df["norm_nutri"] = ((prot_per_eur - mn) / (mx - mn)).round(4).fillna(0) if mx > mn else 0.0

    print("   ✅ texto_busqueda, norm_precio, norm_nutri creados")
    return df

# ---------------------------------------------------------------------------
# Score nutricional
# ---------------------------------------------------------------------------

SCORE_WEIGHTS = {
    "proteinas":     +0.35,
    "fibra":         +0.20,
    "carbohidratos": -0.15,
    "grasas":        -0.15,
    "azucares":      -0.15,
}

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    print("\n✨ ENRIQUECIMIENTO")

    for col in SCORE_WEIGHTS:
        col_data = df[col].fillna(0)
        mn, mx = col_data.min(), col_data.max()
        df[f"_pct_{col}"] = (col_data - mn) / (mx - mn) if mx > mn else 0.5

    def _score(row) -> float:
        s = 0.0
        for col, w in SCORE_WEIGHTS.items():
            pct = row.get(f"_pct_{col}", 0.5)
            s += w * pct * 100 if w > 0 else w * (1 - pct) * 100
        return round(max(0.0, min(100.0, s)), 2)

    df["score_nutricional"] = df.apply(_score, axis=1)
    df = df.drop(columns=[f"_pct_{c}" for c in SCORE_WEIGHTS])

    print(f"   ✅ score_nutricional — "
          f"min={df['score_nutricional'].min():.1f} | "
          f"media={df['score_nutricional'].mean():.1f} | "
          f"max={df['score_nutricional'].max():.1f}")
    return df

# ---------------------------------------------------------------------------
# Columnas finales
# ---------------------------------------------------------------------------

FINAL_COLS = [
    "url", "titulo", "categoria", "origen",
    "precio", "precio_por_kg", "peso_volumen",
    "proteinas", "carbohidratos", "grasas", "fibra",
    "azucares", "saturadas", "sal", "calorias", "calorias_kj",
    "texto_busqueda", "norm_precio", "norm_nutri", "score_nutricional",
]

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in FINAL_COLS if c in df.columns]]

# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"   📁 CSV:  {OUTPUT_CSV}")
    df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)
    print(f"   📁 JSON: {OUTPUT_JSON}")

# ---------------------------------------------------------------------------
# Función pública para main.py
# ---------------------------------------------------------------------------

def procesar_datos(input_path: str = INPUT_FILE) -> pd.DataFrame:
    """
    Pipeline completo: carga → limpieza → normalización → enriquecimiento.
    Devuelve el DataFrame limpio y lo guarda en src/data/clean/.
    """
    print("=" * 60)
    print("PREPROCESSING — Pipeline RAG Ahorramas")
    print("=" * 60)

    df = load_raw(input_path)
    df = clean(df)
    df = normalize(df)
    df = enrich(df)
    df = select_columns(df)

    print("\n📊 RESUMEN FINAL")
    print(f"   Productos totales:            {len(df)}")
    print(f"   Con nutrición completa:       "
          f"{df[['proteinas','carbohidratos','grasas']].notna().all(axis=1).sum()}")
    print(f"   Columnas: {list(df.columns)}")

    print("\n💾 GUARDANDO")
    save(df)

    print("\n✅ Preprocessing completado.")
    print(df[["titulo", "precio", "proteinas", "carbohidratos",
              "grasas", "score_nutricional", "norm_precio", "norm_nutri"]].head(8).to_string())

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_clean = procesar_datos()