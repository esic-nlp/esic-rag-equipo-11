import re, bs4

# Parchea todas las instancias de "lxml" -> "html.parser" en tiempo de ejecución
# (solo necesario si el kernel no tiene lxml instalado)
_orig_init = bs4.BeautifulSoup.__init__

def _patched_init(self, *args, **kwargs):
    if args and len(args) >= 2 and args[1] == "lxml":
        args = (args[0], "html.parser") + args[2:]
    _orig_init(self, *args, **kwargs)

bs4.BeautifulSoup.__init__ = _patched_init
print("Parser parcheado: lxml → html.parser")
"""
acquisition.py
==============
Scraper para supermercado.eroski.es
Extrae productos con información nutricional y guarda en data/raw/eroski_products.json

Uso:
    python acquisition.py

Requisitos:
    pip install requests beautifulsoup4 lxml
"""

import requests
from bs4 import BeautifulSoup
import time
import re
import json
import os
from typing import List, Dict, Optional

print("Imports cargados correctamente")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DOMAIN = "https://supermercado.eroski.es"
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "eroski_products.json")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "es-ES,es;q=0.9"}

session = requests.Session()
session.headers.update(HEADERS)

# ---------------------------------------------------------------------------
# Categorías a scrapear
# ---------------------------------------------------------------------------
URLS: Dict[str, str] = {
    "Arroz": "https://supermercado.eroski.es/es/supermercado/2059806-alimentacion/2060029-legumbres-arroz-y-pasta/2060032-arroz/",
    "Pasta_Macarrones": "https://supermercado.eroski.es/es/supermercado/2059806-alimentacion/2060029-legumbres-arroz-y-pasta/2060034-macarrones-y-pasta-corta/",
    "Pasta_Espagueti": "https://supermercado.eroski.es/es/supermercado/2059806-alimentacion/2060029-legumbres-arroz-y-pasta/2060033-espagueti-tallarines-y-pasta-larga/",
    "Pasta_FideosSopa": "https://supermercado.eroski.es/es/supermercado/2059806-alimentacion/2060029-legumbres-arroz-y-pasta/2060036-fideos-y-pasta-sopa/",
    "Cereales": "https://supermercado.eroski.es/es/supermercado/2060118-dulces-y-desayuno/5000189-cereales-y-barritas/",
    "Leche": "https://supermercado.eroski.es/es/supermercado/2059806-alimentacion/2059807-leche-batidos-y-bebidas-vegetales/",
    "Yogures": "https://supermercado.eroski.es/es/supermercado/2059806-alimentacion/2059818-yogures/",
    "Refrescos": "https://supermercado.eroski.es/es/supermercado/2060211-bebidas/2060219-refrescos/",
    "Aguas": "https://supermercado.eroski.es/es/supermercado/2060211-bebidas/2060212-agua/",
    "Cerveza": "https://supermercado.eroski.es/es/supermercado/2060211-bebidas/2060233-cervezas/",
}

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
PRICE_NOW_RE = re.compile(r"Ahora\s+(\d+[,\.]\d+)\s*€")
PRICE_GENERIC_RE = re.compile(r"(\d+[,\.]\d+)\s*€")
PRICE_PER_RE = re.compile(r"(\d+[,\.]\d+)\s*€\s*/\s*(kg|l|ud|uds|100\s*g|100\s*ml)", re.IGNORECASE)
QTY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g|l|ml|cl|ud|uds)\b", re.IGNORECASE)

# Nutrientes que queremos capturar (clave normalizada → posibles labels en la web)
NUTRIENT_ALIASES: Dict[str, List[str]] = {
    "Valor energetico": ["valor energético", "energia", "energía", "energy", "kcal"],
    "Valor energetico en KJ": ["kj"],
    "Grasas": ["grasas", "grasa total", "fat"],
    "Saturadas": ["saturadas", "saturated", "ácidos grasos saturados"],
    "Monoinsaturadas": ["monoinsaturadas", "monoinsaturated"],
    "Poliinsaturadas": ["poliinsaturadas", "polyunsaturated"],
    "Hidratos de carbono": ["hidratos de carbono", "carbohidratos", "carbohydrate"],
    "Azucares": ["azúcares", "azucares", "sugars"],
    "Polialcoholes": ["polialcoholes", "polyols"],
    "Almidon": ["almidón", "almidon", "starch"],
    "Fibra alimentaria": ["fibra alimentaria", "fibra dietética", "fibra", "fibre", "fiber"],
    "Proteinas": ["proteínas", "proteinas", "protein"],
    "Sal": ["sal", "salt", "sodio", "sodium"],
}


def _norm(text: str) -> str:
    """Normaliza texto para comparaciones: minúsculas, sin tildes extras, strip."""
    return text.lower().strip()


# ---------------------------------------------------------------------------
# Crawler: extrae URLs de producto de páginas de categoría
# ---------------------------------------------------------------------------

def fetch_html(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def extract_product_links(html: str) -> List[str]:
    """Extrae URLs de producto de una página de categoría."""
    soup = BeautifulSoup(html, "lxml")
    seen = set()
    links = []
    for a in soup.select('a[href*="/es/productdetail/"]'):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith("/"):
            href = BASE_DOMAIN + href
        elif not href.startswith("http"):
            continue
        if href not in seen:
            seen.add(href)
            links.append(href)
    return links


def crawl(base_url: str, delay: float = 1.5) -> List[str]:
    """Obtiene todos los enlaces de producto de una URL de categoría."""
    print(f"  Crawling: {base_url}")
    try:
        html = fetch_html(base_url)
    except Exception as e:
        print(f"  ERROR al obtener categoría: {e}")
        return []
    links = extract_product_links(html)
    print(f"  Links encontrados: {len(links)}")
    if not links:
        print("  Aviso: 0 links — puede ser una página de subcategorías.")
    time.sleep(delay)
    return links


# ---------------------------------------------------------------------------
# Scraper: extrae datos del producto desde su página de detalle
# ---------------------------------------------------------------------------

def _parse_price(text_all: str) -> Optional[float]:
    """Devuelve el precio principal como float, o None."""
    m = PRICE_NOW_RE.search(text_all)
    if not m:
        m = PRICE_GENERIC_RE.search(text_all)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _parse_price_per_unit(text_all: str) -> Optional[float]:
    """Devuelve el precio por kg/l/ud, o None."""
    m = PRICE_PER_RE.search(text_all)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _parse_quantity(nombre: Optional[str], text_all: str) -> Optional[str]:
    """Detecta peso/volumen desde el nombre o el texto completo."""
    for source in [nombre or "", text_all]:
        m = QTY_RE.search(source)
        if m:
            return f"{m.group(1)}{m.group(2).lower()}"
    return None


def _parse_nutrition_table(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extrae la tabla nutricional (valores por 100 g/ml).
    Intenta varias estrategias:
      1. <table> con filas label/valor
      2. Secciones <dl> / <dt>/<dd>
      3. Búsqueda de texto libre con regex
    """
    nutrients: Dict[str, str] = {}

    # --- Estrategia 1: buscar tablas ---
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                label = _norm(cells[0].get_text(" ", strip=True))
                value = cells[1].get_text(" ", strip=True).strip()
                _assign_nutrient(nutrients, label, value)

    # --- Estrategia 2: listas de definición ---
    if not nutrients:
        for dl in soup.find_all("dl"):
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                label = _norm(dt.get_text(" ", strip=True))
                value = dd.get_text(" ", strip=True).strip()
                _assign_nutrient(nutrients, label, value)

    # --- Estrategia 3: regex sobre texto plano ---
    if not nutrients:
        text = soup.get_text(" ", strip=True)
        # Busca patrones como "Proteínas 6,7 g" o "Grasas: 30 g"
        nutrient_re = re.compile(
            r"(proteínas?|grasas?|carbohidratos?|hidratos de carbono|azúcares?|fibra|sal|energía|valor energético|sodio)"
            r"[\s:]*(\d+[,\.]?\d*\s*(?:g|mg|kcal|kj|ml)?)",
            re.IGNORECASE
        )
        for m in nutrient_re.finditer(text):
            label = _norm(m.group(1))
            value = m.group(2).strip()
            _assign_nutrient(nutrients, label, value)

    return nutrients


def _assign_nutrient(nutrients: Dict[str, str], label: str, value: str) -> None:
    """Mapea un label crudo a la clave normalizada del diccionario de nutrientes."""
    for canonical, aliases in NUTRIENT_ALIASES.items():
        if canonical in nutrients:
            continue  # ya capturado
        for alias in aliases:
            if alias in label:
                nutrients[canonical] = value
                break


def _parse_description(soup: BeautifulSoup) -> str:
    """Extrae descripción del producto (primer párrafo informativo)."""
    # Intenta encontrar un div o sección de descripción
    for selector in [
        ".product-description", ".productDescription", "#description",
        '[class*="descripcion"]', '[class*="description"]',
    ]:
        el = soup.select_one(selector)
        if el:
            return el.get_text(" ", strip=True)

    # Fallback: primer <p> de longitud razonable
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if 30 < len(txt) < 800:
            return txt
    return ""


def parse_product(html: str, url: str, category: str) -> Dict:
    """
    Parsea la página de un producto y devuelve un dict con el formato objetivo.
    """
    soup = BeautifulSoup(html, "lxml")
    text_all = soup.get_text(" ", strip=True)

    # Título
    nombre_tag = soup.find("h1")
    titulo = nombre_tag.get_text(strip=True) if nombre_tag else None

    # Precios
    precio_total = _parse_price(text_all)
    precio_por_cantidad = _parse_price_per_unit(text_all)

    # Peso / volumen
    peso_volumen = _parse_quantity(titulo, text_all)

    # Información nutricional
    valores_nutricionales = _parse_nutrition_table(soup)

    # Descripción
    descripcion = _parse_description(soup)

    return {
        "url": url,
        "titulo": titulo,
        "valores_nutricionales_100_g": valores_nutricionales,
        "descripcion": descripcion,
        "categorias": [category.lower()],
        "precio_total": precio_total,
        "precio_por_cantidad": precio_por_cantidad,
        "peso_volumen": peso_volumen,
        "origen": "eroski",
    }


def scrape_product(url: str, category: str) -> Optional[Dict]:
    """Obtiene y parsea un producto. Devuelve None si hay error."""
    try:
        html = fetch_html(url)
        return parse_product(html, url, category)
    except Exception as e:
        print(f"    ERROR scrapeando {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main: orquesta crawl + scrape + guardado
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Fase 1: Crawling (recolectar URLs) ---
    print("\n" + "=" * 60)
    print("FASE 1: CRAWLING DE CATEGORÍAS")
    print("=" * 60)

    all_links: Dict[str, List[str]] = {}
    for category, url in URLS.items():
        print(f"\n[{category}]")
        all_links[category] = crawl(url, delay=1.2)

    total_links = sum(len(v) for v in all_links.values())
    print(f"\nTotal URLs de producto encontradas: {total_links}")

    # --- Fase 2: Scraping (extraer datos de cada producto) ---
    print("\n" + "=" * 60)
    print("FASE 2: SCRAPING DE PRODUCTOS")
    print("=" * 60)

    products: List[Dict] = []
    errors = 0

    for category, url_list in all_links.items():
        print(f"\n[{category}] — {len(url_list)} productos")
        for i, url in enumerate(url_list, 1):
            print(f"  ({i}/{len(url_list)}) {url}")
            product = scrape_product(url, category)
            if product:
                products.append(product)
                # Validación mínima de campos obligatorios
                missing = []
                if not product.get("titulo"):
                    missing.append("titulo")
                if product.get("precio_total") is None:
                    missing.append("precio")
                nuts = product.get("valores_nutricionales_100_g", {})
                for req in ["Proteinas", "Hidratos de carbono", "Grasas"]:
                    if req not in nuts:
                        missing.append(req)
                if missing:
                    print(f"    ⚠ Campos faltantes: {missing}")
            else:
                errors += 1
            time.sleep(0.9)

    # --- Fase 3: Guardar JSON ---
    print("\n" + "=" * 60)
    print("FASE 3: GUARDANDO RESULTADOS")
    print("=" * 60)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Productos scrapeados: {len(products)}")
    print(f"❌ Errores: {errors}")
    print(f"📁 JSON guardado en: {OUTPUT_FILE}")

    if len(products) < 200:
        print(f"⚠ ATENCIÓN: solo {len(products)} productos. El requisito mínimo es 200.")
        print("   Considera añadir más categorías en el dict URLS o scrapear sub-categorías.")

    # Resumen por categoría
    print("\nResumen por categoría:")
    from collections import Counter
    cat_count = Counter(p["categorias"][0] for p in products)
    for cat, n in cat_count.most_common():
        print(f"  {cat}: {n}")

    return products


if __name__ == "__main__":
    main()
