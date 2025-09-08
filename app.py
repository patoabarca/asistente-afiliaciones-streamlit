import os, re, json, base64, requests, unicodedata
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st
from openai import OpenAI

# =========================
# CONFIG & BRANDING
# =========================
st.set_page_config(page_title="Asistente de Afiliaciones (v6)", page_icon="🧩", layout="wide")

PALETA_IOMA = {
    "teal":    "#2D8DA6",
    "purpura": "#6A5AAE",
    "magenta": "#C4286F",
    "blanco":  "#FFFFFF",
    "gris":    "#3C3C3C",
}

# CSS suave
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.2rem; }}
h1, h2, h3 {{ color: {PALETA_IOMA['teal']}; }}
div.stButton>button:first-child {{
  background:{PALETA_IOMA['teal']}; color:white; border-radius:10px; border:0;
}}
div.stButton>button[kind="secondary"] {{
  background:{PALETA_IOMA['magenta']}; color:white; border-radius:10px; border:0;
}}
pre.jsonbox {{
  background:#fff; color:{PALETA_IOMA['gris']};
  border:1px solid {PALETA_IOMA['teal']}; border-radius:10px; padding:12px;
  white-space:pre-wrap;
}}
.banner {{
  padding:14px 16px; border-radius:12px;
  background: linear-gradient(90deg, {PALETA_IOMA['teal']}22, {PALETA_IOMA['purpura']}22, {PALETA_IOMA['magenta']}22);
  border:1px solid {PALETA_IOMA['teal']};
}}
.banner h3 {{ margin:0; color:{PALETA_IOMA['teal']}; }}
.banner p  {{ margin:4px 0 0 0; color:{PALETA_IOMA['gris']}; font-size:0.9rem; }}
.small {{ color:{PALETA_IOMA['gris']}; opacity:.7; font-size:.85rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="banner">
  <h3>Asistente de Afiliaciones — v6</h3>
  <p>Consultá requisitos, documentación y buenas prácticas. Las respuestas siguen la BASE vigente y se devuelven en JSON.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# OPENAI CLIENT
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Falta OPENAI_API_KEY. Configurá un Secret en Streamlit Cloud o variable de entorno.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"          # texto
IMAGES_MODEL_PRIMARY  = "dall-e-3"
IMAGES_MODEL_FALLBACK = "gpt-image-1"

# =========================
# DATA
# =========================
DATA_PATH = Path("data/base_conocimiento_afiliaciones_clean.csv")
if not DATA_PATH.exists():
    st.error(f"No encuentro el CSV en {DATA_PATH.resolve()}. Subilo o ajustá la ruta.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df[df["estado"].str.lower().isin(["vigente", "en revisión"])].copy()
    for col in ["id","titulo","contenido","respuesta_validada","palabras_clave"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df

df = load_df(DATA_PATH)

# =========================
# SEARCH (heurística ligera)
# =========================
def dividir_en_tokens(texto: str):
    texto = (texto or "").lower()
    return [t for t in re.split(r"[^a-záéíóúñü0-9]+", texto) if t]

def calcular_relevancia(tokens_consulta, fila):
    puntaje = 0
    puntaje += 3 * len(set(tokens_consulta) & set(dividir_en_tokens(fila.get("palabras_clave",""))))
    puntaje += 2 * len(set(tokens_consulta) & set(dividir_en_tokens(fila.get("titulo",""))))
    puntaje += 1 * len(set(tokens_consulta) & set(dividir_en_tokens(fila.get("contenido",""))))
    return puntaje

def buscar_faq(consulta: str, tabla: pd.DataFrame):
    toks = dividir_en_tokens(consulta)
    puntuadas = [(calcular_relevancia(toks, fila), idx) for idx, fila in tabla.iterrows()]
    puntuadas = [(p,i) for p,i in puntuadas if p>0]
    if not puntuadas:
        return None
    puntuadas.sort(reverse=True)
    return tabla.loc[puntuadas[0][1]]

# =========================
# v6: system + contexto
# =========================
def compactar_texto(s: str, max_chars=800):
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:max_chars]

SYSTEM_PROMPT_V6 = """
Rol: Asistente de Afiliaciones de IOMA. Público: agentes. Tono: institucional.

Instrucciones:
- "checklist": SOLO de <<BASE>>. Si falta dato: "No consta en la normativa adjunta".
- "terminos_clave" y "objetivo_y_buenas_practicas": podés usar conocimiento externo.
- Responder SOLO con JSON válido.

Formato:
{
  "checklist": ["...", "..."],
  "terminos_clave": ["Término: definición breve", "...", "..."],
  "objetivo_y_buenas_practicas": ["Buena práctica: detalle breve", "...", "..."],
  "cierre": "Fuente: base de conocimiento vigente"
}
""".strip()

def build_context_v6(fila, pregunta: str):
    if fila is None:
        base = ""
        idtitulo = "(ninguna)"
    else:
        base = (fila.get("respuesta_validada") or fila.get("contenido") or fila.get("titulo","")).strip()
        base = compactar_texto(base, 800)
        idtitulo = f"{fila.get('id','(sin id)')} – {fila.get('titulo','(sin título)')}"
    payload = f'''
    Pregunta: "{pregunta}"

    <<BASE>>
    {base}
    <<FIN_BASE>>
    '''.strip()
    return payload, idtitulo

# =========================
# IMÁGENES: cache + generación
# =========================
IMGS_DIR = Path("imgs"); IMGS_DIR.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+","-", s).strip("-").lower()
    return s or "imagen"

def resize_to(path_in: Path, path_out: Path, size=(256,256)):
    im = Image.open(path_in).convert("RGB")
    im = im.resize(size, Image.LANCZOS)
    im.save(path_out, format="PNG")
    return path_out

def build_prompt_imagen(tema: str) -> str:
    return f"""
Generar una ilustración institucional clara y elegante sobre **{tema}** como figura/escena central,
con un fondo armónico en degradado de la paleta IOMA (teal {PALETA_IOMA['teal']}, púrpura {PALETA_IOMA['purpura']}, magenta {PALETA_IOMA['magenta']}).
Incluir un halo luminoso o un marco circular sutil detrás del elemento principal para reforzar el foco.
Estilo minimalista, moderno y cálido, sin texto en la imagen. Composición limpia.
"""

def _generate_image_bytes(model: str, prompt_text: str, size: str = "1024x1024"):
    resp = client.images.generate(model=model, prompt=prompt_text, n=1, size=size)
    # URL
    try:
        url = resp.data[0].url
        if url:
            return requests.get(url, timeout=120).content
    except Exception:
        pass
    # b64_json
    try:
        b64 = resp.data[0].b64_json
        if b64:
            return base64.b64decode(b64)
    except Exception:
        pass
    raise RuntimeError("No se pudo obtener la imagen (ni url ni b64_json).")

def get_or_create_image_for_theme(tema: str, size_generate="1024x1024", size_display=(256,256)) -> Path:
    slug = slugify(tema)
    p_full = IMGS_DIR / f"{slug}_1024.png"
    p_disp = IMGS_DIR / f"{slug}_{size_display[0]}x{size_display[1]}.png"

    if p_disp.exists(): return p_disp
    if p_full.exists(): return resize_to(p_full, p_disp, size=size_display)

    prompt_text = build_prompt_imagen(tema)
    try:
        img_bytes = _generate_image_bytes(IMAGES_MODEL_PRIMARY, prompt_text, size_generate)
    except Exception:
        img_bytes = _generate_image_bytes(IMAGES_MODEL_FALLBACK, prompt_text, size_generate)

    Image.open(BytesIO(img_bytes)).convert("RGB").save(p_full)
    resize_to(p_full, p_disp, size=size_display)
    return p_disp

def infer_tema_imagen(consulta: str, fila_sel) -> str:
    # override manual: tema: ...
    m = re.search(r"tema\s*:\s*([^\n\r]+)", consulta, flags=re.IGNORECASE)
    if m:
        override = m.group(1).strip()
        if override:
            return override
    cand = (fila_sel.get("titulo","") if isinstance(fila_sel, dict) else (fila_sel["titulo"] if fila_sel is not None and "titulo" in fila_sel else "")) or ""
    texto_ref = f"{consulta} {cand}".lower()
    rules = [
        (r"reci[eé]n\s*nacid[oa]", "afiliación de recién nacido/a"),
        (r"recien\s*nac",         "afiliación de recién nacido/a"),
        (r"estudiante",           "afiliación de estudiante"),
        (r"conviviente",          "afiliación de conviviente"),
        (r"c[oó]nyuge|conyuge",   "afiliación de cónyuge"),
        (r"monotribut",           "afiliación de monotributista"),
        (r"padre|madre|progenitor","afiliación por vínculo familiar"),
    ]
    for pat, tema in rules:
        if re.search(pat, texto_ref):
            return tema
    return cand.strip() or "afiliaciones IOMA"

# =========================
# CHAT: memoria de conversación
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role":"system", "content": SYSTEM_PROMPT_V6}]
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {"faq_idtitulo": None, "tema": None}

def chat(consulta: str, temperature: float = 0.2, max_tokens: int = 400):
    fila_sel = buscar_faq(consulta, df)
    contexto, idtitulo = build_context_v6(fila_sel, consulta)

    st.session_state.chat_history.append({"role":"user", "content": contexto})
    resp = client.chat.completions.create(
        model=MODEL,
        messages=st.session_state.chat_history,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    assistant_msg = resp.choices[0].message.content
    st.session_state.chat_history.append({"role":"assistant", "content": assistant_msg})

    tema = infer_tema_imagen(consulta, fila_sel.to_dict() if fila_sel is not None else {})
    st.session_state.last_meta = {"faq_idtitulo": idtitulo, "tema": tema}
    return assistant_msg

def reset_history():
    st.session_state.chat_history = [{"role":"system", "content": SYSTEM_PROMPT_V6}]
    st.session_state.last_meta = {"faq_idtitulo": None, "tema": None}

# =========================
# UI
# =========================
colL, colR = st.columns([0.65, 0.35])

with colL:
    with st.form("chat_form", clear_on_submit=False):
        msg = st.text_area("Tu consulta", placeholder='Ej: "recién nacido", "estudiante", "conviviente" (override: tema: estudiante)', height=90)
        c1, c2 = st.columns([0.3, 0.7])
        send = c1.form_submit_button("Enviar")
        reset = c2.form_submit_button("Reset historial", type="secondary")

    if reset:
        reset_history()
        st.success("Historial reiniciado.")

    if send:
        if not msg.strip():
            st.warning("Escribí un mensaje primero.")
        else:
            st.markdown(f"**Usuario:** {msg}")
            ans = chat(msg)

            faq_label = st.session_state.last_meta.get("faq_idtitulo") or "(ninguna)"
            st.markdown(f'<div class="small">FAQ seleccionada: {faq_label}</div>', unsafe_allow_html=True)
            st.markdown("**Asistente**")
            st.markdown(f'<pre class="jsonbox">{ans}</pre>', unsafe_allow_html=True)

with colR:
    tema = (st.session_state.last_meta or {}).get("tema") or "afiliaciones IOMA"
    try:
        path_256 = get_or_create_image_for_theme(tema, size_generate="1024x1024", size_display=(256,256))
        st.image(str(path_256), caption=f"Imagen: {tema}", width=256)
    except Exception as e:
        st.info(f"No se pudo mostrar la imagen ({e}).")

st.markdown('<div class="small">Tip: podés forzar el tema con <code>tema: ...</code>. Las imágenes se cachean en <code>imgs/</code>.</div>', unsafe_allow_html=True)
