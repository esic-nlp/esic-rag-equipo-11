"""
main.py
=======
Punto de entrada del pipeline RAG Nutricional — NutriSearch.

Flujo completo:
    1. acquisition.py  → scraping de Ahorramas → data/raw/ahorramas_products.json
    2. preprocessing.py → limpieza y normalización → data/clean/ahorramas_products_clean.csv
    3. rag.py           → embeddings FAISS + interfaz web en http://localhost:5000

Uso:
    python main.py
"""

import sys
import os

# Añade src/ al path para que funcionen los imports
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAW_JSON = os.path.join(SRC, "data", "raw", "ahorramas_products.json")
CLEAN_CSV = os.path.join(SRC, "data", "clean", "ahorramas_products_clean.csv")


def main():
    print("=" * 60)
    print("  🛒  NUTRISEARCH — Pipeline RAG Nutricional")
    print("=" * 60)

    # ── Paso 1: Acquisition ──────────────────────────────────────────────────
    print("\n🕷️  Paso 1: Recogiendo datos de Ahorramas...")
    print("   (esto puede tardar varios minutos)")

    from acquisition import main as run_acquisition
    run_acquisition()
    print("   ✅ Datos scrapeados correctamente.")

    # ── Paso 2: Preprocessing ────────────────────────────────────────────────
    print("\n📦 Paso 2: Preprocesando datos...")

    # Cambiamos el directorio de trabajo a src/ para que las rutas relativas
    # del preprocessing.py funcionen correctamente
    os.chdir(SRC)

    from preprocessing import load_raw_json, preprocess_products, save_dataframe, OUTPUT_FILE
    raw_data = load_raw_json(os.path.join("data", "raw", "ahorramas_products.json"))
    df = preprocess_products(raw_data)
    save_dataframe(df, OUTPUT_FILE)
    print(f"   ✅ {len(df)} productos procesados.")

    # El rag.py espera norm_nutri — el preprocessing del repo lo genera como norm_nutri ✅
    # Pero necesita texto_busqueda ✅ y norm_precio ✅ — todo OK

    # ── Paso 3: RAG + interfaz web ───────────────────────────────────────────
    print("\n🌐 Paso 3: Lanzando interfaz web en http://localhost:5000 ...")

    # Volvemos a la raíz
    os.chdir(ROOT)

    from src.rag import consultar
    consultar(df)


if __name__ == "__main__":
    main()
