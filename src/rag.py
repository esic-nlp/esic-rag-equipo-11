"""
rag.py
======
Sistema RAG con interfaz web local (Flask).
Abre automáticamente http://localhost:5000 al ejecutarse.

Requisitos adicionales:
    pip install flask
"""

import os
import json
import threading
import webbrowser

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Modelo global
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------------------------------------------------------------------------
# Cocinas del mundo → keywords para filtrar por categoría culinaria
# ---------------------------------------------------------------------------
COCINAS = {
    "🇮🇹 Italiana":   ["pasta", "macarrones", "espagueti", "penne", "lasaña", "risotto",
                        "pizza", "fideo", "tallarín", "gnocchi", "ravioli"],
    "🇲🇽 Mexicana":   ["taco", "tortilla", "jalapeño", "frijol", "guacamole", "burrito",
                        "nachos", "chile", "maíz"],
    "🇺🇸 Americana":  ["hamburguesa", "hot dog", "bacon", "cheddar", "bbq", "nugget",
                        "ketchup", "mayonesa", "patatas fritas", "cola", "refresco"],
    "🇯🇵 Japonesa":   ["sushi", "soja", "miso", "wasabi", "arroz", "alga", "sake",
                        "ramen", "teriyaki", "tofu"],
    "🇬🇷 Mediterránea": ["aceite oliva", "aceite de oliva", "atún", "sardina", "anchoa",
                          "aceituna", "yogur", "queso", "tomate", "albahaca"],
    "🇮🇳 India":      ["curry", "cúrcuma", "garam", "masala", "lentejas", "chickpea",
                        "garbanzo", "basmati", "cardamomo"],
    "🇨🇳 China":      ["salsa soja", "wok", "bambú", "jengibre", "fideos", "dim sum",
                        "bok choy", "sésamo"],
    "🥗 Saludable":   ["proteina", "fibra", "integral", "ecológico", "sin gluten",
                        "bio", "light", "0%", "sin azúcar", "vegano", "vegetal"],
}

# ---------------------------------------------------------------------------
# Core RAG
# ---------------------------------------------------------------------------

def crear_indice(df: pd.DataFrame):
    print("[RAG] Generando embeddings e índice FAISS...")
    embeddings = embedder.encode(df["texto_busqueda"].tolist(), show_progress_bar=False)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype("float32"))
    print(f"[RAG] Índice creado con {len(df)} productos.")
    return index


def _detectar_atributos(consulta: str) -> list:
    q = consulta.lower()
    mapa = {
        "proteinas":     ["proteina", "proteínas", "músculo", "musculo"],
        "grasas":        ["grasa", "light", "bajo en grasa"],
        "calorias":      ["caloria", "caloría", "bajo en calorías"],
        "sal":           ["sal", "sodio", "sin sal"],
        "precio":        ["barato", "económico", "precio", "oferta"],
        "salud":         ["sano", "saludable", "healthy", "nutritivo"],
        "carbohidratos": ["hidrato", "carbohidrato", "energía", "energia"],
        "fibra":         ["fibra", "digestivo", "digestión"],
    }
    return [k for k, kws in mapa.items() if any(kw in q for kw in kws)]


def _detectar_cocina(consulta: str) -> str | None:
    q = consulta.lower()
    for cocina, kws in COCINAS.items():
        if any(kw in q for kw in kws):
            return cocina
    return None


def buscar_productos(consulta: str, df: pd.DataFrame, index,
                     cocina_filtro: str = None,
                     max_precio: float = None,
                     min_proteinas: float = None,
                     solo_con_nutri: bool = False,
                     n_resultados: int = 8) -> list:
    """
    Busca y rankea productos. Devuelve lista de dicts listos para JSON.
    """
    atributos = _detectar_atributos(consulta)

    # Búsqueda vectorial
    vec = embedder.encode([consulta]).astype("float32")
    dist, idx = index.search(vec, min(50, len(df)))

    cands = df.iloc[idx[0]].copy()
    max_d = dist[0].max() if dist[0].max() > 0 else 1
    cands["norm_dist"] = 1 - (dist[0] / max_d)

    # Score final base
    cands["rank_final"] = (
        cands["norm_dist"]   * 0.6 +
        cands["norm_nutri"]  * 0.2 +
        cands["norm_precio"] * 0.2
    )

    # Boost por atributos detectados
    if "proteinas" in atributos:
        p_max = cands["proteinas"].max() or 1
        cands["rank_final"] += (cands["proteinas"].fillna(0) / p_max) * 0.3
    if "precio" in atributos:
        cands["rank_final"] += cands["norm_precio"] * 0.3
    if "salud" in atributos:
        cands["rank_final"] += (cands["score_nutricional"].fillna(0) / 100) * 0.3
    if "fibra" in atributos:
        f_max = cands["fibra"].max() or 1
        cands["rank_final"] += (cands["fibra"].fillna(0) / f_max) * 0.2

    # Filtros opcionales
    if cocina_filtro and cocina_filtro in COCINAS:
        kws = COCINAS[cocina_filtro]
        mask = cands["texto_busqueda"].apply(
            lambda t: any(kw in str(t).lower() for kw in kws)
        )
        if mask.sum() > 0:
            cands = cands[mask]

    if max_precio is not None:
        cands = cands[cands["precio"] <= max_precio]

    if min_proteinas is not None:
        cands = cands[cands["proteinas"].fillna(0) >= min_proteinas]

    if solo_con_nutri:
        cands = cands[cands[["proteinas", "carbohidratos", "grasas"]].notna().all(axis=1)]

    mejores = cands.sort_values("rank_final", ascending=False).head(n_resultados)

    def _fmt(v, dec=1):
        try:
            return round(float(v), dec) if v == v else None  # NaN check
        except Exception:
            return None

    results = []
    for _, r in mejores.iterrows():
        results.append({
            "titulo":           str(r.get("titulo", "")).title(),
            "precio":           _fmt(r.get("precio"), 2),
            "precio_por_kg":    _fmt(r.get("precio_por_kg"), 2),
            "peso_volumen":     str(r.get("peso_volumen", "") or ""),
            "categoria":        str(r.get("categoria", "") or ""),
            "proteinas":        _fmt(r.get("proteinas")),
            "carbohidratos":    _fmt(r.get("carbohidratos")),
            "grasas":           _fmt(r.get("grasas")),
            "fibra":            _fmt(r.get("fibra")),
            "calorias":         _fmt(r.get("calorias"), 0),
            "sal":              _fmt(r.get("sal")),
            "score_nutricional": _fmt(r.get("score_nutricional"), 0),
            "url":              str(r.get("url", "") or ""),
            "origen":           str(r.get("origen", "") or ""),
            "rank":             _fmt(r.get("rank_final"), 3),
        })
    return results


# ---------------------------------------------------------------------------
# HTML de la interfaz web
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🛒 NutriSearch — Asistente Nutricional</title>
<style>
  :root {
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --card2: #16213e;
    --accent: #e94560;
    --accent2: #0f3460;
    --green: #00d4aa;
    --yellow: #ffd700;
    --text: #e0e0e0;
    --muted: #888;
    --radius: 16px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 24px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 2px solid var(--accent);
    box-shadow: 0 4px 30px rgba(233,69,96,0.2);
  }
  header h1 { font-size: 1.8rem; font-weight: 800; letter-spacing: -1px; }
  header h1 span { color: var(--accent); }
  header p { font-size: 0.85rem; color: var(--muted); margin-top: 2px; }
  .logo { font-size: 2.4rem; }

  /* ── Layout ── */
  main { display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 90px); }

  /* ── Sidebar ── */
  aside {
    background: var(--card);
    padding: 24px 20px;
    border-right: 1px solid #2a2a3e;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }
  aside h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: var(--muted); }

  /* Cocinas */
  .cocina-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .cocina-btn {
    background: var(--card2);
    border: 2px solid transparent;
    border-radius: 12px;
    padding: 10px 6px;
    cursor: pointer;
    text-align: center;
    font-size: 0.75rem;
    color: var(--text);
    transition: all 0.2s;
    line-height: 1.3;
  }
  .cocina-btn:hover { border-color: var(--accent); background: #1e1e3a; }
  .cocina-btn.active { border-color: var(--accent); background: rgba(233,69,96,0.15); color: #fff; }
  .cocina-emoji { font-size: 1.4rem; display: block; margin-bottom: 4px; }

  /* Filtros */
  .filtro-group { display: flex; flex-direction: column; gap: 8px; }
  .filtro-group label { font-size: 0.8rem; color: var(--muted); }
  .filtro-group input[type=range] { width: 100%; accent-color: var(--accent); }
  .filtro-group .val { font-size: 0.85rem; color: var(--green); font-weight: 600; }
  .filtro-group input[type=number] {
    background: var(--card2);
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 8px 12px;
    color: var(--text);
    font-size: 0.85rem;
    width: 100%;
  }
  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.82rem;
  }
  .toggle {
    position: relative;
    width: 40px;
    height: 22px;
  }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .slider-toggle {
    position: absolute;
    inset: 0;
    background: #2a2a3e;
    border-radius: 22px;
    cursor: pointer;
    transition: 0.3s;
  }
  .slider-toggle::before {
    content: '';
    position: absolute;
    width: 16px; height: 16px;
    left: 3px; top: 3px;
    background: white;
    border-radius: 50%;
    transition: 0.3s;
  }
  .toggle input:checked + .slider-toggle { background: var(--accent); }
  .toggle input:checked + .slider-toggle::before { transform: translateX(18px); }

  /* ── Search area ── */
  .content {
    padding: 28px 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }
  .search-box {
    display: flex;
    gap: 12px;
    align-items: center;
  }
  .search-box input {
    flex: 1;
    background: var(--card);
    border: 2px solid #2a2a3e;
    border-radius: var(--radius);
    padding: 16px 20px;
    font-size: 1rem;
    color: var(--text);
    transition: border-color 0.2s;
    outline: none;
  }
  .search-box input:focus { border-color: var(--accent); }
  .search-box input::placeholder { color: var(--muted); }
  .btn-search {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: var(--radius);
    padding: 16px 28px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }
  .btn-search:hover { background: #c73652; transform: translateY(-1px); }
  .btn-search:active { transform: translateY(0); }

  /* Sugerencias rápidas */
  .suggestions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .sug-chip {
    background: var(--card2);
    border: 1px solid #2a2a3e;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--muted);
  }
  .sug-chip:hover { border-color: var(--accent); color: var(--text); background: #1e1e3a; }

  /* ── Resultados ── */
  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .results-header h2 { font-size: 1rem; color: var(--muted); }
  .results-header .count { color: var(--accent); font-weight: 700; }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
  }

  .product-card {
    background: var(--card);
    border-radius: var(--radius);
    padding: 20px;
    border: 1px solid #2a2a3e;
    transition: all 0.25s;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }
  .product-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--green));
    opacity: 0;
    transition: opacity 0.2s;
  }
  .product-card:hover { transform: translateY(-4px); border-color: #3a3a5e; box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
  .product-card:hover::before { opacity: 1; }

  .card-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 14px; }
  .card-title { font-size: 0.9rem; font-weight: 700; line-height: 1.3; flex: 1; }
  .card-price {
    background: rgba(233,69,96,0.15);
    color: var(--accent);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.9rem;
    font-weight: 800;
    white-space: nowrap;
  }

  .card-cat {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }

  /* Score bar */
  .score-bar-wrap { margin-bottom: 14px; }
  .score-label { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }
  .score-label span:last-child { color: var(--green); font-weight: 700; }
  .score-bar { height: 6px; background: #2a2a3e; border-radius: 3px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent2), var(--green)); transition: width 0.6s ease; }

  /* Macros mini */
  .macros {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 4px;
  }
  .macro {
    background: var(--card2);
    border-radius: 8px;
    padding: 6px 4px;
    text-align: center;
  }
  .macro-val { font-size: 0.85rem; font-weight: 700; color: var(--text); }
  .macro-lbl { font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .macro.prot .macro-val { color: var(--green); }
  .macro.carb .macro-val { color: var(--yellow); }
  .macro.fat  .macro-val { color: #ff8c42; }

  .card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #2a2a3e;
    font-size: 0.72rem;
    color: var(--muted);
  }
  .origen-badge {
    background: var(--accent2);
    color: #aac;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Loading */
  .loading {
    display: none;
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
  }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid #2a2a3e;
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Empty */
  .empty {
    display: none;
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
    font-size: 1rem;
  }
  .empty .emoji { font-size: 3rem; margin-bottom: 12px; }

  /* Responsive */
  @media (max-width: 768px) {
    main { grid-template-columns: 1fr; }
    aside { border-right: none; border-bottom: 1px solid #2a2a3e; }
    .content { padding: 20px 16px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">🥗</div>
  <div>
    <h1>Nutri<span>Search</span></h1>
    <p>Asistente nutricional inteligente · Ahorramas</p>
  </div>
</header>

<main>
  <!-- Sidebar con filtros -->
  <aside>
    <div>
      <h2>🌍 Cocina del mundo</h2>
      <div class="cocina-grid" id="cocinaGrid">
        <!-- generado por JS -->
      </div>
    </div>

    <div>
      <h2>🎛️ Filtros</h2>
      <div class="filtro-group">
        <label>Precio máximo: <span class="val" id="precioVal">sin límite</span></label>
        <input type="range" id="maxPrecio" min="0" max="50" step="0.5" value="50"
               oninput="updatePrecio(this.value)">
      </div>
      <div class="filtro-group">
        <label>Mínimo de proteínas (g/100g)</label>
        <input type="number" id="minProteinas" placeholder="ej: 10" min="0" max="100" step="1">
      </div>
      <div class="filtro-group">
        <div class="toggle-row">
          <span>Solo productos con nutrición</span>
          <label class="toggle">
            <input type="checkbox" id="soloNutri">
            <span class="slider-toggle"></span>
          </label>
        </div>
      </div>
    </div>

    <div>
      <h2>⚡ Búsquedas rápidas</h2>
      <div style="display:flex;flex-direction:column;gap:6px">
        <button class="sug-chip" style="text-align:left" onclick="quickSearch('alto en proteínas y barato')">💪 Alto en proteínas</button>
        <button class="sug-chip" style="text-align:left" onclick="quickSearch('bajo en grasas y saludable')">🥦 Bajo en grasas</button>
        <button class="sug-chip" style="text-align:left" onclick="quickSearch('para desayuno con fibra')">🌅 Desayuno con fibra</button>
        <button class="sug-chip" style="text-align:left" onclick="quickSearch('económico y nutritivo')">💰 Económico y nutritivo</button>
        <button class="sug-chip" style="text-align:left" onclick="quickSearch('pasta italiana')">🍝 Pasta italiana</button>
      </div>
    </div>
  </aside>

  <!-- Contenido principal -->
  <section class="content">
    <div class="search-box">
      <input type="text" id="queryInput" placeholder="¿Qué estás buscando? ej: 'algo rico en proteínas para después del gym'"
             onkeydown="if(event.key==='Enter') buscar()">
      <button class="btn-search" onclick="buscar()">🔍 Buscar</button>
    </div>

    <div class="suggestions">
      <span style="font-size:0.8rem;color:var(--muted);align-self:center">Sugerencias:</span>
      <span class="sug-chip" onclick="setQuery(this)">🍗 pollo</span>
      <span class="sug-chip" onclick="setQuery(this)">🥛 leche sin lactosa</span>
      <span class="sug-chip" onclick="setQuery(this)">🐟 atún en aceite</span>
      <span class="sug-chip" onclick="setQuery(this)">🌾 cereales integrales</span>
      <span class="sug-chip" onclick="setQuery(this)">🧀 queso bajo en grasa</span>
      <span class="sug-chip" onclick="setQuery(this)">🍺 cerveza</span>
    </div>

    <div class="loading" id="loading">
      <div class="spinner"></div>
      <p>Buscando los mejores productos...</p>
    </div>

    <div class="empty" id="empty">
      <div class="emoji">🔍</div>
      <p>No encontramos productos. Prueba con otros filtros.</p>
    </div>

    <div id="resultsHeader" class="results-header" style="display:none">
      <h2>Resultados: <span class="count" id="countBadge">0</span> productos</h2>
      <span id="cocinaActive" style="font-size:0.85rem;color:var(--muted)"></span>
    </div>

    <div class="results-grid" id="resultsGrid"></div>
  </section>
</main>

<script>
const COCINAS = ["🇮🇹 Italiana","🇲🇽 Mexicana","🇺🇸 Americana","🇯🇵 Japonesa",
                 "🇬🇷 Mediterránea","🇮🇳 India","🇨🇳 China","🥗 Saludable"];

let cocinaSeleccionada = null;

// Renderizar botones de cocina
const grid = document.getElementById("cocinaGrid");
COCINAS.forEach(c => {
  const parts = c.split(" ");
  const emoji = parts[0];
  const name  = parts.slice(1).join(" ");
  const btn = document.createElement("button");
  btn.className = "cocina-btn";
  btn.innerHTML = `<span class="cocina-emoji">${emoji}</span>${name}`;
  btn.onclick = () => toggleCocina(c, btn);
  grid.appendChild(btn);
});

function toggleCocina(cocina, btn) {
  if (cocinaSeleccionada === cocina) {
    cocinaSeleccionada = null;
    btn.classList.remove("active");
  } else {
    document.querySelectorAll(".cocina-btn").forEach(b => b.classList.remove("active"));
    cocinaSeleccionada = cocina;
    btn.classList.add("active");
  }
  if (document.getElementById("queryInput").value.trim()) buscar();
}

function updatePrecio(v) {
  document.getElementById("precioVal").textContent = v >= 50 ? "sin límite" : v + "€";
}

function setQuery(el) {
  document.getElementById("queryInput").value = el.textContent.replace(/^[^\w]+/, "").trim();
  buscar();
}

function quickSearch(q) {
  document.getElementById("queryInput").value = q;
  buscar();
}

function scoreColor(s) {
  if (s >= 70) return "#00d4aa";
  if (s >= 40) return "#ffd700";
  return "#e94560";
}

async function buscar() {
  const query = document.getElementById("queryInput").value.trim();
  if (!query && !cocinaSeleccionada) return;

  const maxP  = parseFloat(document.getElementById("maxPrecio").value);
  const minPr = parseFloat(document.getElementById("minProteinas").value) || null;
  const soloN = document.getElementById("soloNutri").checked;

  document.getElementById("loading").style.display = "block";
  document.getElementById("resultsGrid").innerHTML  = "";
  document.getElementById("resultsHeader").style.display = "none";
  document.getElementById("empty").style.display    = "none";

  try {
    const res = await fetch("/buscar", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        consulta: query || (cocinaSeleccionada ? "comida " + cocinaSeleccionada : ""),
        cocina: cocinaSeleccionada,
        max_precio: maxP >= 50 ? null : maxP,
        min_proteinas: minPr,
        solo_con_nutri: soloN
      })
    });

    const data = await res.json();
    document.getElementById("loading").style.display = "none";

    if (!data.resultados || data.resultados.length === 0) {
      document.getElementById("empty").style.display = "block";
      return;
    }

    document.getElementById("resultsHeader").style.display = "flex";
    document.getElementById("countBadge").textContent = data.resultados.length;
    document.getElementById("cocinaActive").textContent =
      cocinaSeleccionada ? "Filtro: " + cocinaSeleccionada : "";

    const grid = document.getElementById("resultsGrid");
    data.resultados.forEach((p, i) => {
      const score = p.score_nutricional ?? 0;
      const hasNutri = p.proteinas !== null || p.carbohidratos !== null || p.grasas !== null;

      const card = document.createElement("div");
      card.className = "product-card";
      card.style.animationDelay = (i * 0.05) + "s";
      card.onclick = () => { if (p.url) window.open(p.url, "_blank"); };

      card.innerHTML = `
        <div class="card-header">
          <div class="card-title">${p.titulo}</div>
          <div class="card-price">${p.precio != null ? p.precio + "€" : "—"}</div>
        </div>
        <div class="card-cat">${p.categoria.replace(/_/g," ")} · ${p.origen}</div>

        ${hasNutri ? `
        <div class="score-bar-wrap">
          <div class="score-label">
            <span>Score nutricional</span>
            <span>${score}/100</span>
          </div>
          <div class="score-bar">
            <div class="score-fill" style="width:${score}%;background:linear-gradient(90deg,#0f3460,${scoreColor(score)})"></div>
          </div>
        </div>
        <div class="macros">
          <div class="macro prot">
            <div class="macro-val">${p.proteinas != null ? p.proteinas + "g" : "—"}</div>
            <div class="macro-lbl">Proteínas</div>
          </div>
          <div class="macro carb">
            <div class="macro-val">${p.carbohidratos != null ? p.carbohidratos + "g" : "—"}</div>
            <div class="macro-lbl">Carbos</div>
          </div>
          <div class="macro fat">
            <div class="macro-val">${p.grasas != null ? p.grasas + "g" : "—"}</div>
            <div class="macro-lbl">Grasas</div>
          </div>
        </div>` : `<div style="color:var(--muted);font-size:0.78rem;margin:8px 0">Sin información nutricional</div>`}

        <div class="card-footer">
          <span>${p.peso_volumen || ""} ${p.calorias ? "· " + p.calorias + " kcal" : ""}</span>
          <span class="origen-badge">${p.origen}</span>
        </div>
      `;
      grid.appendChild(card);
    });

  } catch(e) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("empty").style.display = "block";
    console.error(e);
  }
}

// Buscar al cargar si hay query params
window.onload = () => {
  document.getElementById("queryInput").focus();
};
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def crear_app(df: pd.DataFrame, index):
    app = Flask(__name__)
    app.df    = df
    app.index = index

    @app.route("/")
    def home():
        return HTML

    @app.route("/buscar", methods=["POST"])
    def buscar():
        body          = request.get_json(force=True)
        consulta      = body.get("consulta", "")
        cocina        = body.get("cocina")
        max_precio    = body.get("max_precio")
        min_prot      = body.get("min_proteinas")
        solo_nutri    = body.get("solo_con_nutri", False)

        resultados = buscar_productos(
            consulta      = consulta,
            df            = app.df,
            index         = app.index,
            cocina_filtro = cocina,
            max_precio    = float(max_precio) if max_precio is not None else None,
            min_proteinas = float(min_prot)   if min_prot   is not None else None,
            solo_con_nutri= solo_nutri,
        )
        return jsonify({"resultados": resultados})

    return app


# ---------------------------------------------------------------------------
# Función pública para main.py
# ---------------------------------------------------------------------------

def consultar(df: pd.DataFrame, host: str = "127.0.0.1", port: int = 5000):
    """
    Crea el índice FAISS, lanza el servidor Flask y abre el navegador.
    Llamada desde main.py: rag.consultar(df)
    """
    index = crear_indice(df)
    app   = crear_app(df, index)

    url = f"http://{host}:{port}"
    print(f"\n🌐 Abriendo NutriSearch en {url}")
    print("   Pulsa Ctrl+C para detener el servidor.\n")

    # Abrir navegador tras 1 segundo (para dar tiempo a Flask)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Main standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE, "data", "clean", "products_clean.csv")
    df = pd.read_csv(csv_path)
    consultar(df)
