"""
main.py
=======
Punto de entrada del pipeline RAG Nutricional.

Uso:
    python main.py
"""

import sys
import os

# Añade la raíz al path para que funcionen los imports de src.*
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.preprocessing import procesar_datos
from ragggg import consultar


def main():
    print("=" * 60)
    print("  🛒  NUTRISEARCH — Pipeline RAG Nutricional")
    print("=" * 60)

    # 1. Preprocessing
    print("\n📦 Paso 1: Preprocesando datos...")
    df = procesar_datos()
    print(f"   ✅ {len(df)} productos listos.")

    # 2. RAG + Web
    print("\n🌐 Paso 2: Lanzando interfaz web...")
    consultar(df)


if __name__ == "__main__":
    main()
