"""
preprocessing.py
================
Limpia, transforma y enriquece los datos de productos scrapeados
para su uso en el pipeline RAG.

Entrada:  data/raw/eroski_products.json
Salida:   data/clean/products_clean.csv  +  data/clean/products_clean.parquet

Uso:
    python preprocessing.py
"""

import json
import os
import re
import pandas as pd
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
INPUT_FILE  = os.path.join("data", "raw", "eroski_products.json")
OUTPUT_DIR  = os.path.join("data", "clean")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "products_clean.csv")
OUTPUT_PARQ = os.path.join(OUTPUT_DIR, "products_clean.parquet")

# ---------------------------------------------------------------------------
# Helpers de parseo numérico
# ---------------------------------------------------------------------------

def _to_float(value) -> Optional[float]:
    """Convierte strings tipo '6,7 g', '537 kcal', 30.4, None → float o None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if not np.isnan(float(value)) else None
    # Extraer primer número del string
    m = re.search(r"(\d+[.,]?\d*)", str(value).replace(",", "."))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _clean_text(text) -> str:
    """Minusculiza, elimina espacios extra y caracteres raros."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáéíóúüñ,.()/%-]", "", text)
    return text


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    print(f"📂 Cargando: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    print(f"   Registros cargados: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Extracción de nutrientes desde el dict anidado
# ---------------------------------------------------------------------------

NUTRIENT_MAP = {
    # columna destino            posibles claves en valores_nutricionales_100_g
    "calorias":      ["Valor energetico", "Energia", "Energía"],
    "calorias_kj":   ["Valor energetico en KJ"],
    "grasas":        ["Grasas"],
    "saturadas":     ["Saturadas"],
    "carbohidratos": ["Hidratos de carbono", "Carbohidratos"],
    "azucares":      ["Azucares", "Azúcares"],
    "fibra":         ["Fibra alimentaria", "Fibra"],
    "proteinas":     ["Proteinas", "Proteínas"],
    "sal":           ["Sal"],
}

# Columna con el dict nutricional tras json_normalize
NUTRI_COL = "valores_nutricionales_100_g"


def extract_nutrients(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae cada nutriente como columna float independiente."""
    # json_normalize puede dejar el dict como objeto o expandirlo con prefijo
    # Detectamos cuál es el caso
    if NUTRI_COL in df.columns:
        nutri_series = df[NUTRI_COL]
    else:
        # json_normalize lo expandió con prefijo "valores_nutricionales_100_g."
        prefix = NUTRI_COL + "."
        nutri_cols = [c for c in df.columns if c.startswith(prefix)]
        if nutri_cols:
            # Reconstruimos el dict por fila
            nutri_series = df[nutri_cols].rename(
                columns=lambda c: c.replace(prefix, "")
            ).apply(lambda row: row.dropna().to_dict(), axis=1)
            df = df.drop(columns=nutri_cols)
        else:
            nutri_series = pd.Series([{}] * len(df), index=df.index)

    for col_name, keys in NUTRIENT_MAP.items():
        def _extract(d, keys=keys):
            if not isinstance(d, dict):
                return None
            for k in keys:
                if k in d:
                    return _to_float(d[k])
            return None
        df[col_name] = nutri_series.apply(_extract)

    return df


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------

CRITICAL_FIELDS = ["titulo", "precio_total", "proteinas", "carbohidratos", "grasas"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 LIMPIEZA")
    n0 = len(df)

    # 1. Extraer nutrientes
    df = extract_nutrients(df)

    # 2. Renombrar precio
    if "precio_total" in df.columns:
        df = df.rename(columns={"precio_total": "precio"})
    if "precio_por_cantidad" in df.columns:
        df = df.rename(columns={"precio_por_cantidad": "precio_por_kg"})

    # 3. Convertir tipos numéricos
    for col in ["precio", "precio_por_kg", "calorias", "calorias_kj",
                "grasas", "saturadas", "carbohidratos", "azucares",
                "fibra", "proteinas", "sal"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    # 4. Eliminar filas sin campos críticos
    # Ajustamos CRITICAL_FIELDS a los que realmente existen
    existing_critical = [f if f != "precio_total" else "precio" for f in CRITICAL_FIELDS]
    existing_critical = [f for f in existing_critical if f in df.columns]
    df = df.dropna(subset=existing_critical)
    print(f"   Eliminadas por campos críticos nulos: {n0 - len(df)}")

    # 5. Eliminar duplicados por URL y por título
    n1 = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["titulo"], keep="first")
    print(f"   Eliminados duplicados: {n1 - len(df)}")

    # 6. Limpiar textos
    df["titulo"] = df["titulo"].apply(_clean_text)
    if "descripcion" in df.columns:
        df["descripcion"] = df["descripcion"].apply(_clean_text)
    else:
        df["descripcion"] = ""

    print(f"   Registros tras limpieza: {len(df)}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📐 NORMALIZACIÓN")

    # --- texto_busqueda ---
    marca = df["origen"] if "origen" in df.columns else ""
    df["texto_busqueda"] = (
        df["titulo"].fillna("") + " " +
        marca.fillna("") + " " +
        df["descripcion"].fillna("") + " " +
        df["categorias"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        ).fillna("")
    ).str.strip().apply(_clean_text)

    # --- norm_precio (0-1, inverso: más barato → valor más alto) ---
    p = df["precio"]
    p_min, p_max = p.min(), p.max()
    if p_max > p_min:
        df["norm_precio"] = 1 - (p - p_min) / (p_max - p_min)
    else:
        df["norm_precio"] = 0.5
    df["norm_precio"] = df["norm_precio"].round(4)

    # --- norm_nutri (0-1 basado en proteínas por euro) ---
    # Proteínas / precio → normalizado a 0-1
    prot_per_eur = df["proteinas"] / df["precio"].replace(0, np.nan)
    mn, mx = prot_per_eur.min(), prot_per_eur.max()
    if mx > mn:
        df["norm_nutri"] = ((prot_per_eur - mn) / (mx - mn)).round(4)
    else:
        df["norm_nutri"] = 0.5
    df["norm_nutri"] = df["norm_nutri"].fillna(0)

    print("   ✅ texto_busqueda, norm_precio, norm_nutri creados")
    return df


# ---------------------------------------------------------------------------
# Enriquecimiento: score_nutricional
# ---------------------------------------------------------------------------

# Pesos del score (suman 100)
SCORE_WEIGHTS = {
    "proteinas":     0.35,   # +  queremos alto
    "fibra":         0.20,   # +  queremos alto
    "carbohidratos": -0.15,  # -  penalizamos exceso
    "grasas":        -0.15,  # -  penalizamos exceso
    "azucares":      -0.15,  # -  penalizamos exceso
}


def _score_nutricional(row) -> float:
    """
    Score de 0-100 basado en macronutrientes.
    Cada nutriente se puntúa relativo a los percentiles de la columna
    (ya calculados antes de aplicar esta función).
    """
    score = 0.0
    for col, weight in SCORE_WEIGHTS.items():
        val = row.get(f"_pct_{col}", 0.5)
        if weight > 0:
            score += weight * val * 100
        else:
            score += weight * (1 - val) * 100   # invertimos para negativos
    return round(max(0, min(100, score)), 2)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    print("\n✨ ENRIQUECIMIENTO")

    # Precalcular percentiles por columna para el score
    for col in SCORE_WEIGHTS:
        if col in df.columns:
            col_clean = df[col].fillna(0)
            mn, mx = col_clean.min(), col_clean.max()
            if mx > mn:
                df[f"_pct_{col}"] = (col_clean - mn) / (mx - mn)
            else:
                df[f"_pct_{col}"] = 0.5
        else:
            df[f"_pct_{col}"] = 0.5

    df["score_nutricional"] = df.apply(_score_nutricional, axis=1)

    # Eliminar columnas auxiliares de percentil
    df = df.drop(columns=[f"_pct_{c}" for c in SCORE_WEIGHTS])

    # Categoría como string limpio
    if "categorias" in df.columns:
        df["categoria"] = df["categorias"].apply(
            lambda x: x[0] if isinstance(x, list) and x else str(x)
        )

    print(f"   ✅ score_nutricional generado (media: {df['score_nutricional'].mean():.1f})")
    return df


# ---------------------------------------------------------------------------
# Selección de columnas finales
# ---------------------------------------------------------------------------

FINAL_COLS = [
    "url", "titulo", "categoria", "origen",
    "precio", "precio_por_kg", "peso_volumen",
    "proteinas", "carbohidratos", "grasas", "fibra",
    "azucares", "saturadas", "sal", "calorias", "calorias_kj",
    "texto_busqueda", "norm_precio", "norm_nutri", "score_nutricional",
]


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in FINAL_COLS if c in df.columns]
    return df[existing]


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    try:
        df.to_parquet(OUTPUT_PARQ, index=False)
        print(f"   📁 Parquet: {OUTPUT_PARQ}")
    except ImportError:
        print("   ⚠ pyarrow no instalado, solo se guarda CSV.")
    print(f"   📁 CSV:     {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    print("=" * 60)
    print("PREPROCESSING — Pipeline RAG Eroski")
    print("=" * 60)

    df = load_raw(INPUT_FILE)
    df = clean(df)
    df = normalize(df)
    df = enrich(df)
    df = select_columns(df)

    print("\n📊 RESUMEN FINAL")
    print(f"   Productos limpios: {len(df)}")
    print(f"   Columnas: {list(df.columns)}")
    print(f"\n   Nulos por columna crítica:")
    for col in ["proteinas", "carbohidratos", "grasas", "precio"]:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"     {col}: {n} nulos ({n/len(df)*100:.1f}%)")

    print(f"\n   Score nutricional — min: {df['score_nutricional'].min():.1f} "
          f"| media: {df['score_nutricional'].mean():.1f} "
          f"| max: {df['score_nutricional'].max():.1f}")

    print("\n💾 GUARDANDO")
    save(df)

    print("\n✅ Preprocessing completado.")
    print(df[["titulo", "precio", "proteinas", "carbohidratos",
              "grasas", "score_nutricional", "norm_precio", "norm_nutri"]].head(5).to_string())

    return df


if __name__ == "__main__":
    df_clean = main()