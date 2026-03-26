"""
acquisition_ahorramas.py
========================
Scraper para www.ahorramas.com
Genera data/raw/ahorramas_products.json con el formato estándar del proyecto.

CÓMO FUNCIONA LA NUTRICIÓN:
  Ahorramas muestra la tabla nutricional como una IMAGEN en el carrusel del producto.
  La primera imagen (código C1C1 en la URL) es la foto frontal.
  La segunda imagen (código EIAh u otro) es la tabla nutricional/ingredientes.
  Este script descarga TODAS las imágenes del carrusel y aplica OCR a cada una
  hasta extraer los nutrientes.

REQUISITOS:
  pip install requests beautifulsoup4 pytesseract Pillow
  Windows → instalar Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
            y añadir al PATH, o descomentar la línea tesseract_cmd más abajo.
"""

import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io, re
import time
import re
import json
import os
import io
from collections import Counter
from typing import List, Dict, Optional

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
    # En Windows, descomenta y ajusta si tesseract no está en el PATH
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    OCR_AVAILABLE = False
    print("pytesseract o Pillow no están instalados. Instala con: pip install pytesseract Pillow")


BASE_DOMAIN = "https://www.ahorramas.com"
OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ahorramas_products.json")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "es-ES,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.ahorramas.com/",
}

URLS: Dict[str, str] = {
    "Alimentacion_Ofertas": "https://www.ahorramas.com/ofertas/alimentacion/",
    "Bebidas_Ofertas": "https://www.ahorramas.com/ofertas/bebidas/",
    "Congelados_Ofertas": "https://www.ahorramas.com/ofertas/congelados/",
    "Frescos_Ofertas": "https://www.ahorramas.com/ofertas/frescos/",
    "Carniceria": "https://www.ahorramas.com/frescos/carniceria/",
    "Pollo": "https://www.ahorramas.com/frescos/carniceria/pollo/",
    "Charcuteria": "https://www.ahorramas.com/frescos/charcuteria/",
    "Ternera_y_vacuno": "https://www.ahorramas.com/frescos/carniceria/ternera-y-vacuno/",
    "Guisantes_Congelados": "https://www.ahorramas.com/congelados/verduras-hortalizas-y-frutas-congeladas/guisantes/",
    "Embutidos_ibericos": "https://www.ahorramas.com/frescos/charcuteria/embutidos-ibericos/",
    "Fiambres_y_cocidos": "https://www.ahorramas.com/frescos/charcuteria/fiambres-y-cocidos/",
    "Empanados_y_Elaborados": "https://www.ahorramas.com/frescos/carniceria/empanados-y-elaborados/",
    "Aceite_Oliva_Virgen": "https://www.ahorramas.com/alimentacion/aceite-vinagre-y-sal/aceites/aceite-de-oliva-virgen-y-virgen-extra/",
    "Arroz_Redondo": "https://www.ahorramas.com/alimentacion/arroces-pastas-y-legumbres/arroz/grano-redondo/",
    "Pasta_Corta": "https://www.ahorramas.com/alimentacion/arroces-pastas-y-legumbres/pasta/macarrones-y-pasta-corta/",
    "Legumbres_Lentejas": "https://www.ahorramas.com/alimentacion/arroces-pastas-y-legumbres/legumbres/lentejas/",
    "Conservas_Atun": "https://www.ahorramas.com/alimentacion/conservas-de-pescado/atun/",
    "Galletas_Maria": "https://www.ahorramas.com/alimentacion/galletas-cereales-y-barritas/galletas/galletas-maria/",
    "Cereales_Chocolate": "https://www.ahorramas.com/alimentacion/galletas-cereales-y-barritas/cereales/cereales-con-chocolate/",
    "Cafe_Capsulas": "https://www.ahorramas.com/alimentacion/cacao-cafes-e-infusiones/cafes/capsulas-de-cafe/",
    "Quesos_Semicurados": "https://www.ahorramas.com/frescos/quesos/quesos-semicurados/",
    "Pescado_Blanco": "https://www.ahorramas.com/frescos/pescado-y-mariscos/merluza-y-otro-pescado-blanco/",
    "Huevos_Camperos": "https://www.ahorramas.com/frescos/huevos/camperos-y-ecologicos/",
    "Refrescos_Cola": "https://www.ahorramas.com/bebidas/refrescos/de-cola/",
    "Cervezas_Rubias": "https://www.ahorramas.com/bebidas/cerveza/cervezas-rubias/",
    "Zumos_Naranja": "https://www.ahorramas.com/bebidas/zumos/naranja/",
    "Pizzas_Congeladas": "https://www.ahorramas.com/congelados/pizzas-y-baguettes/pizzas/",
    "Helados_Bombon": "https://www.ahorramas.com/congelados/helados/helado-bombon/",
}

QTY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g|l|ml|cl|ud|uds)\b", re.IGNORECASE)
PRICE_RE = re.compile(r"(\d+[,\.]\d+)\s*€")
PER_RE = re.compile(r"(\d+[,\.]\d+)\s*€\s*/\s*(kg|l|100\s*g|100\s*ml|ud)", re.IGNORECASE)
KJ_RE = re.compile(r"(\d+[.,]?\d*)\s*k[jJ]")
KCAL_RE = re.compile(r"(\d+[.,]?\d*)\s*k?cal", re.IGNORECASE)
VAL_RE = re.compile(r"(\d+[.,]\d+|\d+)\s*(g|mg)?")

CANONICAL: Dict[str, str] = {
    "valor energético": "Valor energetico",
    "valor energetico": "Valor energetico",
    "energía": "Valor energetico",
    "energia": "Valor energetico",
    "calorías": "Valor energetico",
    "calorias": "Valor energetico",
    "valor energético en kj": "Valor energetico en KJ",
    "valor energetico en kj": "Valor energetico en KJ",
    "grasas": "Grasas",
    "grasa total": "Grasas",
    "materia grasa": "Grasas",
    "lípidos": "Grasas",
    "lipidos": "Grasas",
    "ácidos grasos saturados": "Saturadas",
    "grasas saturadas": "Saturadas",
    "saturadas": "Saturadas",
    "de las cuales saturadas": "Saturadas",
    "- saturadas": "Saturadas",
    "ácidos grasos monoinsaturados": "Monoinsaturadas",
    "monoinsaturadas": "Monoinsaturadas",
    "ácidos grasos poliinsaturados": "Poliinsaturadas",
    "poliinsaturadas": "Poliinsaturadas",
    "hidratos de carbono": "Hidratos de carbono",
    "carbohidratos": "Hidratos de carbono",
    "azúcares": "Azucares",
    "azucares": "Azucares",
    "- azúcares": "Azucares",
    "- azucares": "Azucares",
    "de los cuales azúcares": "Azucares",
    "de los cuales azucares": "Azucares",
    "polialcoholes": "Polialcoholes",
    "fibra alimentaria": "Fibra alimentaria",
    "fibra dietética": "Fibra alimentaria",
    "fibra": "Fibra alimentaria",
    "proteínas": "Proteinas",
    "proteinas": "Proteinas",
    "sal": "Sal",
    "sodio": "Sal",
}

NUTRI_KEYWORDS = [
    "energético",
    "energetico",
    "proteín",
    "proteina",
    "grasa",
    "carbono",
    "fibra",
    "kcal",
    "azúcar",
]



# Indica la ruta completa al ejecutable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def sacarinfo(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
        "Accept-Language": "es-ES,es;q=0.9",
    }

    url = "https://www.ahorramas.com/patatas-fritas-lays-170g-gourmet-76305.html"
    resp = requests.get(url, headers=headers, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Todas las imágenes del producto con TODOS sus atributos
    print("=== TODAS LAS IMÁGENES (src + data-src + data-lazy) ===")
    for img in soup.find_all("img"):
        src      = img.get("src", "")
        data_src = img.get("data-src", "")
        lazy     = img.get("data-lazy-src", "")
        alt      = img.get("alt", "")
        cls      = img.get("class", "")
        if src or data_src or lazy:
            print(f"  alt='{alt}' class={cls}")
            print(f"    src={src[:80]}")
            print(f"    data-src={data_src[:80]}")
            print(f"    data-lazy={lazy[:80]}")
            print()

    # Buscar si hay un div/section de "ficha técnica" o "información"
    print("\n=== SECCIONES CON CLASES RELEVANTES ===")
    for tag in soup.find_all(True, class_=True):
        cls = " ".join(tag.get("class", []))
        if any(k in cls.lower() for k in ["detail", "tab", "ficha", "info", "product", "slide", "carousel"]):
            imgs = tag.find_all("img")
            if imgs:
                print(f"  <{tag.name} class='{cls}'> — {len(imgs)} imágenes")

def imagenes():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
        "Accept-Language": "es-ES,es;q=0.9",
    }

    url = "https://www.ahorramas.com/patatas-fritas-lays-170g-gourmet-76305.html"
    resp = requests.get(url, headers=headers, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Las imágenes sin alt y sin clase son las de Library (tabla nutricional)
    print("=== IMÁGENES SIN ALT (candidatas a tabla nutricional) ===")
    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "")
        if not alt and "Library-Sites" in src:
            print(f"  {src}")

    # Ver URLs completas del carrusel del producto
    print("\n=== IMÁGENES DEL CARRUSEL (zoom-img / img-fluid) ===")
    for img in soup.select(".carousel-item img"):
        print(f"  alt='{img.get('alt','')}' src={img.get('src','')}")

def selector():

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
        "Accept-Language": "es-ES,es;q=0.9",
    }

    session = requests.Session()
    session.headers.update(headers)

    url = "https://www.ahorramas.com/patatas-fritas-lays-170g-gourmet-76305.html"
    resp = session.get(url, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Ver EXACTAMENTE qué imágenes encuentra el selector
    print("=== .carousel-item img ===")
    for img in soup.select(".carousel-item img"):
        src = img.get("src", "")
        print(f"  src={src}")

    print("\n=== Probando OCR en cada imagen del carrusel ===")
    for img in soup.select(".carousel-item img"):
        src = img.get("src", "")
        if not src or "dw/image" not in src:
            continue
        # Pedir imagen grande
        src_large = re.sub(r"sh=\d+", "sh=800", src)
        src_large = re.sub(r"sw=\d+", "sw=800", src_large)
        print(f"\nDescargando: {src_large[:80]}...")
        try:
            r = session.get(src_large, timeout=15)
            img_obj = Image.open(io.BytesIO(r.content)).convert("L")
            text = pytesseract.image_to_string(img_obj, lang="spa", config="--psm 6")
            print(f"OCR output:\n{text[:400]}")
            print("---")
        except Exception as e:
            print(f"ERROR: {e}")

# #########################################################################
# Código hasta celda 3
# #########################################################################




try:
    OCR_AVAILABLE = True
    # Windows: descomenta y ajusta si tesseract no está en el PATH
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    OCR_AVAILABLE = False
    print("⚠ pytesseract/Pillow no instalados. Instala con: pip install pytesseract Pillow")

print("Imports cargados correctamente")
print(f"OCR disponible: {OCR_AVAILABLE}")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DOMAIN = "https://www.ahorramas.com"
OUTPUT_DIR  = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ahorramas_products.json")


def _norm(t: str) -> str:
    return t.lower().strip()

def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    m = re.search(r"(\d+[.,]\d+|\d+)", str(val))
    if m:
        return float(m.group(1).replace(",", "."))
    return None

def _assign(nutrients: Dict, label_raw: str, value: str) -> None:
    key = _norm(label_raw)
    canonical = CANONICAL.get(key)
    if canonical and canonical not in nutrients and value:
        nutrients[canonical] = value.strip()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _ocr_bytes(img_bytes: bytes) -> str:
    """Aplica OCR a bytes de imagen, devuelve texto extraído."""
    if not OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        w, h = img.size
        # Escalar si es pequeña (mínimo 800px de ancho para buena precisión)
        if w < 800:
            scale = 800 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return pytesseract.image_to_string(img, lang="spa", config="--psm 6")
    except Exception:
        return ""


def _is_nutrition_text(text: str) -> bool:
    """Comprueba si el texto OCR contiene palabras clave nutricionales."""
    t = text.lower()
    return sum(1 for k in NUTRI_KEYWORDS if k in t) >= 2


def _parse_ocr_nutrition(text: str) -> Dict[str, str]:
    """
    Parsea el texto OCR de la tabla nutricional.
    La tabla tiene columnas: Label | 100g | porción | %IR
    Solo capturamos la columna de 100g (primer valor numérico de cada fila).
    """
    nutrients: Dict[str, str] = {}
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]
        line_lower = _norm(line)

        # Valor energético: puede estar en 2 líneas (kJ y kcal)
        if "energ" in line_lower:
            kj_m   = KJ_RE.search(line)
            kcal_m = KCAL_RE.search(line)
            # kcal puede estar en la línea siguiente
            if kj_m and not kcal_m and i + 1 < len(lines):
                kcal_m = KCAL_RE.search(lines[i + 1])
            if kj_m:
                nutrients["Valor energetico en KJ"] = f"{kj_m.group(1).replace(',','.')} kJ"
            if kcal_m:
                nutrients["Valor energetico"] = f"{kcal_m.group(1).replace(',','.')} kcal"
            i += 1
            continue

        # Resto de nutrientes: buscar label conocido + primer valor numérico
        for label_key, canonical in CANONICAL.items():
            if canonical in ("Valor energetico", "Valor energetico en KJ"):
                continue
            if label_key in line_lower:
                val_m = VAL_RE.search(line)
                if val_m and canonical not in nutrients:
                    val  = val_m.group(1).replace(",", ".")
                    unit = val_m.group(2) or "g"
                    nutrients[canonical] = f"{val} {unit}"
                break
        i += 1

    return nutrients


def _get_carousel_image_urls(soup: BeautifulSoup) -> List[str]:
    """
    Extrae todas las URLs de imagen del carrusel del producto.
    En Ahorramas, las imágenes del producto tienen:
      - alt = nombre del producto (no vacío)
      - class que contiene 'img-fluid' o 'zoom-img'
      - src con dominio dw/image/v2/BFNH_PRD
    La foto frontal tiene código C1C1 en la URL.
    La tabla nutricional tiene otro código (EIAh, etc.) — es la 2ª imagen.
    """
    seen: set = set()
    urls: List[str] = []

    # Selector específico: imágenes dentro de carousel-item con src de producto
    for img in soup.select(".carousel-item img"):
        src = img.get("src", "").strip()
        if src and src not in seen and "dw/image/v2" in src:
            seen.add(src)
            urls.append(src)

    # Fallback: cualquier img con clase img-fluid y src de producto
    if not urls:
        for img in soup.find_all("img", class_=lambda c: c and "img-fluid" in c):
            src = img.get("src", "").strip()
            if src and src not in seen and "dw/image/v2" in src:
                seen.add(src)
                urls.append(src)

    return urls


def nutrition_from_ocr(soup: BeautifulSoup, session: requests.Session) -> Dict[str, str]:
    """
    Descarga TODAS las imágenes del carrusel y aplica OCR a cada una.
    La primera imagen (C1C1) es la foto del producto — normalmente sin texto nutricional.
    La segunda (EIAh u otro código) suele ser la tabla nutricional.
    Probamos todas hasta encontrar la que contenga nutrientes.
    """
    if not OCR_AVAILABLE:
        return {}

    img_urls = _get_carousel_image_urls(soup)
    if not img_urls:
        return {}

    # Pedimos versión grande de la imagen (mejor OCR):
    # Cambiamos parámetros de tamaño en la URL de Salesforce CC
    def _large_url(url: str) -> str:
        # Reemplaza sh=450&sw=400 por sh=800&sw=800
        url = re.sub(r"sh=\d+", "sh=800", url)
        url = re.sub(r"sw=\d+", "sw=800", url)
        return url

    for img_url in img_urls:
        img_url = _large_url(img_url)
        try:
            resp = session.get(img_url, timeout=15)
            resp.raise_for_status()
            text = _ocr_bytes(resp.content)
            if not text or not _is_nutrition_text(text):
                continue
            nutrients = _parse_ocr_nutrition(text)
            if nutrients:
                return nutrients
        except Exception:
            continue

    return {}


# ---------------------------------------------------------------------------
# Extracción nutricional HTML (fallback antes de OCR)
# ---------------------------------------------------------------------------

def _nutrition_from_jsonld(soup: BeautifulSoup) -> Dict[str, str]:
    nutrients: Dict[str, str] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        for item in (data if isinstance(data, list) else [data]):
            if not isinstance(item, dict):
                continue
            ni = item.get("nutrition") or {}
            if not isinstance(ni, dict):
                continue
            for sk, canonical in {
                "calories": "Valor energetico", "fatContent": "Grasas",
                "saturatedFatContent": "Saturadas", "carbohydrateContent": "Hidratos de carbono",
                "sugarContent": "Azucares", "fiberContent": "Fibra alimentaria",
                "proteinContent": "Proteinas", "sodiumContent": "Sal",
            }.items():
                val = ni.get(sk)
                if val and canonical not in nutrients:
                    nutrients[canonical] = str(val)
    return nutrients


def _nutrition_from_table(soup: BeautifulSoup) -> Dict[str, str]:
    nutrients: Dict[str, str] = {}
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                _assign(nutrients,
                        cells[0].get_text(" ", strip=True),
                        cells[-1].get_text(" ", strip=True))
    return nutrients


def _nutrition_from_dl(soup: BeautifulSoup) -> Dict[str, str]:
    nutrients: Dict[str, str] = {}
    for dl in soup.find_all("dl"):
        for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
            _assign(nutrients, dt.get_text(" ", strip=True), dd.get_text(" ", strip=True))
    return nutrients


def parse_nutrition(soup: BeautifulSoup, session: requests.Session) -> Dict[str, str]:
    """Cascada: JSON-LD → tabla HTML → dl → OCR sobre imágenes del carrusel."""
    for fn in [
        lambda: _nutrition_from_jsonld(soup),
        lambda: _nutrition_from_table(soup),
        lambda: _nutrition_from_dl(soup),
        lambda: nutrition_from_ocr(soup, session),
    ]:
        result = fn()
        if result:
            return result
    return {}


# ---------------------------------------------------------------------------
# Resto de parsers
# ---------------------------------------------------------------------------

def parse_alergenos(soup: BeautifulSoup) -> List[str]:
    alergenos = []
    for sel in ["#alergenos", ".alergenos", "[class*='alergen']", "[class*='allergen']"]:
        section = soup.select_one(sel)
        if section:
            for item in section.find_all(["li", "span", "p", "div"]):
                t = item.get_text(strip=True)
                if t and len(t) < 100:
                    alergenos.append(t)
            if alergenos:
                return list(dict.fromkeys(alergenos))
    text = soup.get_text(" ", strip=True)
    m = re.search(r"[Aa]l[eé]rgenos?\s*:?\s*([^.]{5,300})", text)
    if m:
        items = re.split(r"[,;]|\sy\s", m.group(1).strip())
        alergenos = [i.strip() for i in items if 2 < len(i.strip()) < 80]
    return list(dict.fromkeys(alergenos))


def parse_precio(soup: BeautifulSoup, text_all: str) -> Optional[float]:
    meta = soup.find("meta", itemprop="price")
    if meta:
        val = _to_float(meta.get("content", ""))
        if val:
            return val
    for sel in [".price .value[content]", ".sales .value[content]",
                "[itemprop='price']", ".price-sales",
                ".product-price .price", ".pdp-price .value",
                ".price .sales .value", ".price span.value"]:
        el = soup.select_one(sel)
        if el:
            content = el.get("content") or el.get_text(strip=True)
            val = _to_float(content)
            if val and val > 0:
                return val
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict):
                offers = item.get("offers") or {}
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                price = offers.get("price") if isinstance(offers, dict) else None
                if price:
                    return _to_float(price)
    m = PRICE_RE.search(text_all)
    if m:
        return _to_float(m.group(1))
    return None


def parse_precio_por_unidad(text_all: str) -> Optional[float]:
    m = PER_RE.search(text_all)
    return _to_float(m.group(1)) if m else None


def parse_titulo(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True).upper()
    og = soup.find("meta", property="og:title")
    if og:
        return og.get("content", "").strip().upper()
    return None


def parse_peso_volumen(titulo: str, text_all: str) -> Optional[str]:
    for source in [titulo or "", text_all]:
        m = QTY_RE.search(source)
        if m:
            return f"{m.group(1)}{m.group(2).lower()}"
    return None


def parse_descripcion(soup: BeautifulSoup) -> str:
    for sel in [".product-description", ".short-description",
                "[itemprop='description']", ".description", "#description"]:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(" ", strip=True)
            if len(text) > 10:
                return text
    return ""


def parse_direccion_manufactura(soup: BeautifulSoup) -> List[str]:
    result = []
    text = soup.get_text(" ", strip=True)
    addr_re = re.compile(
        r"((?:c/|calle|av\.|avda\.|avenida|polígono|pol\.)[^.\n]{5,80})",
        re.IGNORECASE
    )
    for m in addr_re.finditer(text):
        addr = m.group(1).strip()
        if addr not in result:
            result.append(addr)
    return result[:3]


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def fetch_html(url: str, session: requests.Session, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def extract_product_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set = set()
    links: List[str] = []
    for sel in ["div[data-pid] a.link", ".product-tile a.pdp-link",
                "a.pdp-link", ".product-name a", ".product-tile a[href]"]:
        for a in soup.select(sel):
            href = a.get("href", "").strip()
            if not href:
                continue
            if not href.startswith("http"):
                href = BASE_DOMAIN + href
            if re.search(r"-\d+\.html$", href) and href not in seen:
                seen.add(href)
                links.append(href)
        if links:
            break
    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href.startswith("http"):
                href = BASE_DOMAIN + href
            if re.search(r"ahorramas\.com/.+-\d+\.html$", href) and href not in seen:
                seen.add(href)
                links.append(href)
    return links


def crawl(base_url: str, delay: float = 1.5) -> List[str]:
    print(f"  Crawling: {base_url}")
    try:
        html = fetch_html(base_url)
    except Exception as e:
        print(f"  ERROR: {e}")
        return []
    links = extract_product_links(html)
    print(f"  Links encontrados: {len(links)}")
    time.sleep(delay)
    return links


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def parse_product(html: str, url: str, category: str, session: requests.Session) -> Dict:
    soup = BeautifulSoup(html, "html.parser")
    text_all = soup.get_text(" ", strip=True)
    titulo = parse_titulo(soup)
    return {
        "url":                         url,
        "titulo":                      titulo,
        "valores_nutricionales_100_g": parse_nutrition(soup, session),
        "descripcion":                 parse_descripcion(soup),
        "categorias":                  [category.lower()],
        "precio_total":                parse_precio(soup, text_all),
        "precio_por_cantidad":         parse_precio_por_unidad(text_all),
        "peso_volumen":                parse_peso_volumen(titulo, text_all),
        "alergenos":                   parse_alergenos(soup),
        "origen":                      "ahorramas",
        "direccion_manufactura":       parse_direccion_manufactura(soup),
    }


def scrape_product(url: str, category: str, session: requests.Session) -> Optional[Dict]:
    try:
        html = fetch_html(url, session)
        return parse_product(html, url, category, session)
    except Exception as e:
        print(f"    ERROR scrapeando {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> List[Dict]:
    
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
    HEADERS = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.ahorramas.com/",
    }
    session = requests.Session()
    session.headers.update(HEADERS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("FASE 1: CRAWLING DE CATEGORÍAS")
    print("=" * 60)

    all_links: Dict[str, List[str]] = {}
    for category, url in URLS.items():
        print(f"\n[{category}]")
        all_links[category] = crawl(url, delay=1.5)

    total_links = sum(len(v) for v in all_links.values())
    print(f"\nTotal URLs encontradas: {total_links}")

    print("\n" + "=" * 60)
    print("FASE 2: SCRAPING DE PRODUCTOS")
    print("=" * 60)

    products: List[Dict] = []
    errors   = 0
    seen_urls: set = set()

    for category, url_list in all_links.items():
        print(f"\n[{category}] — {len(url_list)} productos")
        for i, url in enumerate(url_list, 1):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            print(f"  ({i}/{len(url_list)}) {url}")
            product = scrape_product(url, category, session)
            if product:
                products.append(product)
                nuts = product["valores_nutricionales_100_g"]
                missing = [r for r in ["Proteinas", "Hidratos de carbono", "Grasas"]
                           if r not in nuts]
                if not product["precio_total"]:
                    missing.append("precio")
                if missing:
                    print(f"    ⚠ Faltantes: {missing}")
                else:
                    print(f"     {list(nuts.keys())}")
            else:
                errors += 1
            time.sleep(0.9)

    print("\n" + "=" * 60)
    print("FASE 3: GUARDANDO RESULTADOS")
    print("=" * 60)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    total_ok = sum(1 for p in products if p["valores_nutricionales_100_g"])
    print(f"\n Productos scrapeados:        {len(products)}")
    print(f" Con información nutricional: {total_ok} ({total_ok/max(len(products),1)*100:.0f}%)")
    print(f" Errores:                      {errors}")
    print(f" JSON guardado en:             {OUTPUT_FILE}")

    from collections import Counter
    print("\nResumen por categoría:")
    for cat, n in Counter(p["categorias"][0] for p in products).most_common():
        nutri_n = sum(1 for p in products
                      if p["categorias"][0] == cat and p["valores_nutricionales_100_g"])
        print(f"  {cat}: {n} ({nutri_n} con nutrición)")

    if products:
        ejemplo = next((p for p in products if p["valores_nutricionales_100_g"]), products[0])
        print("\nEjemplo:")
        print(json.dumps(ejemplo, ensure_ascii=False, indent=2))

    return products


if __name__ == "__main__":
    sacarinfo()
    imagenes()
    selector()
    main()