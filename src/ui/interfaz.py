import streamlit as st
import sys
import os
import asyncio
from pathlib import Path
import httpx

# Add src to python path to allow imports from api and utilidades
sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.api import app_graph
from utilidades import utils

# Ruta base de imágenes (relativa al proyecto) para no depender de rutas absolutas de otro PC
IMAGENES_DIR = Path(__file__).resolve().parents[2] / "data" / "documentos" / "imagenes"

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Autónomos Bizkaia",
    page_icon="🍁",
    layout="wide",
)

ST_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #3B1E1E; /* Slate 800 */
    }
    
    #MainMenu, footer {visibility: hidden;}
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 100 !important;
    }
    
    .block-container {
        padding-bottom: 2rem !important;
        padding-top: 2rem !important;
    }

    .stApp {
        background-color: #FFFFFF; /* Slate 50 */
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(246 59 59 / 0.05) 0%, transparent 50%);
        background-attachment: fixed;
    }

    .title {
        max-width: 700px !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 auto !important;
    }

    h1 {
        background: linear-gradient(135deg, #000000   0%, #000000  80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 250 !important;
        font-size: 2.5rem !important;
        margin-bottom: -1.5rem !important;
        letter-spacing: -0.5px;
        margin-top: 7rem !important;
    }
    h2 {
        background: linear-gradient(135deg, #000000  0%, #000000 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 500 !important;
        font-size: 3rem !important;
        margin-top: -1.5rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.5px;
    }
    h4 {
        background: linear-gradient(135deg, #000000  0%, #000000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 300 !important;
        font-size: 2rem !important;
        letter-spacing: -0.5px;
        margin-top: -1.5rem !important;
        text-align: center !important;
    }
    h5 {
        background: linear-gradient(135deg, #FF4343  0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 350 !important;
        font-size: 1rem !important;
        margin-top: -1.5rem !important;
        letter-spacing: -0.5px;
        text-align: center !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFF6F6 !important;
        border-right: 1px solid #F0E2E2; /* Borde gris claro */
        box-shadow: 2px 0 10px rgba(0,0,0,0.03); /* Sombra muy ligera */
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2A0F0F !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #694747;
    }

    .user-bubble {
        background-color: #FFF6F6;
        color: #3B1E1E;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #FFC9C9;
    }

    
    .ai-bubble {
        background-color: #FFF6F6;
        color: #553333;
        padding: 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 15px 0;
        max-width: 85%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #FFC9C9;
        border-left: 2px solid #FF6767;
    }
    
    .ai-bubble .content p {
        margin: 0.5rem 0;
        line-height: 1.6;
        color: #553333;
    }
    
    .ai-bubble .content ul, 
    .ai-bubble .content ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        color: #553333;
    }
    
    .ai-bubble .content li {
        margin: 0.25rem 0;
        color: #553333;
    }
    
    .ai-bubble .content strong {
        color: #2A0F0F;
        font-weight: 600;
    }
    
    .ai-bubble .content em {
        color: #694747;
    }
    
    .ai-bubble .content h1, 
    .ai-bubble .content h2, 
    .ai-bubble .content h3 {
        color: #2A0F0F;
        margin: 0.75rem 0 0.5rem 0;
    }
    
    .sources-container {
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid #FFC9C9;
    }

    /* Evita que burbujas sin texto ocupen espacio o muestren bordes */
    .ai-bubble:empty, .user-bubble:empty {
        display: none !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }
    
    /* INPUT FLOTANTE (ADAPTIVE) - Ahora en columnas */
    .stChatInput {
        bottom: 40px !important;
        background: transparent !important;
    }
    
    div[data-testid="stChatInput"] {
        /* Chat input dentro de columna - sin max-width aquí */
        background: #FFFFFF !important;
        border-radius: 0.6rem !important;
        border: 2px solid #E1CBCB !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* File uploader más compacto y bonito */
    .stFileUploader {
        margin-top: 0 !important;
    }
    
    .stFileUploader > div {
        padding: 0.5rem !important;
        border-radius: 0.6rem !important;
        border: 2px solid #E1CBCB !important;
        background: #FFFFFF !important;
        min-height: 60px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stFileUploader label {
        font-size: 1.5rem !important;
        text-align: center !important;
        display: block !important;
        margin: 0 !important;
    }
    
    .stFileUploader section {
        border: none !important;
    }

    .source-chip {
        display: inline-flex;
        align-items: center;
        background: #FFF0F0;
        border: 1px solid #FDBABA;
        color: #A10303;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 8px;
        text-decoration: none;
        transition: transform 0.1s ease;
    }
    
    .source-chip:hover {
        background: #FEE0E0;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    @keyframes pulseBorder {
        0% { 
            box-shadow: 0 0 0 0 rgba(246 59 59 / 0.4); 
            border-color: #FD9393;
        }
        70% { 
            box-shadow: 0 0 0 12px rgba(246 59 59 / 0); 
            border-color: #F63B3B;
        }
        100% { 
            box-shadow: 0 0 0 0 rgba(246 59 59 / 0); 
            border-color: #FD9393;
        }
    }

    .thinking-bubble {
        background-color: #FFF6F6;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 15px 0;
        width: fit-content;
        border: 2px solid #F63B3B;
        animation: pulseBorder 1.5s infinite ease-in-out;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .thinking-dots {
        display: flex;
        gap: 6px;
    }
    .dot {
        width: 8px;
        height: 8px;
        background: #F63B3B;
        border-radius: 50%;
        animation: simpleBounce 1s infinite ease-in-out both;
    }
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes simpleBounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    .image-caption {
        padding: 0.5rem;
        font-size: 0.75rem;
        color: #694747;
        background: #FFF8F8;
    }

    /* Galería de Imágenes */
    .image-gallery {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #FFC9C9;
    }
    
    .image-gallery-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #694747;
        margin-bottom: 0.75rem;
    }
    
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 0.5rem;
    }
    
    .image-card {
        border: 1px solid #FFC9C9;
        border-radius: 8px;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        background: #FFFFFF;
    }
    
    .image-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .image-card img {
        width: 100%;
        height: 150px;
        object-fit: cover;
        display: block;
    }

    /* ESTILOS LANDING GEMINI */
    .landing-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        width: 100%;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }

    .landing-title {
        margin-bottom: 0.5rem !important;
        text-align: left;
        width: 100%;
        max-width: 700px;
        margin-left: 0 !important;
    }

    .landing-title h1 {
        margin-top: 0 !important;
        margin-bottom: -3rem !important;
        line-height: 1.2;
        margin-left: 0 !important;
    }

    .landing-title h2 {
        margin-top: 0 !important;
        line-height: 1.2;
    }

    .suggestions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmin(180px, 1fr));
        gap: 1rem;
        max-width: 700px; /* Ajustado de 800px a 700px */
        width: 100%;
        margin-top: 2rem;
        padding: 0; /* Eliminado padding lateral */
    }

    /* Estilo para los botones que actúan como tarjetas */
    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid #E1CBCB !important;
        background-color: #FFF6F6 !important;
        color: #3B1E1E !important;
        height: auto !important;
        min-height: 45px !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
        width: 100% !important; /* Asegurar que ocupe el ancho de la columna */
    }

    .stButton > button:hover {
        border-color: #F96A6A !important;
        background-color: #FFF6F6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(246, 59, 59, 0.08) !important;
        color: #A10303 !important;
    }

    .stButton > button p {
        font-size: 0.85rem !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
</style>
"""
st.markdown(ST_STYLE, unsafe_allow_html=True)

def get_base64_encoded_image(image_path):
    import base64
    try:
        if not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return None

def render_message(role, content, sources=None, imagenes=None):
    """Renderiza un mensaje con mejor estructura y soporte para imágenes."""
    if not content or not content.strip():
        return

    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        # Burbuja de la IA - Texto
        st.markdown(f'''
        <div class="ai-bubble">
            <div class="content">
                {content.replace("**", "<b>")}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Galería de imágenes usando componentes nativos para mayor estabilidad
        if imagenes and len(imagenes) > 0:
            st.markdown("---")
            st.markdown("**Imágenes relacionadas**")
            
            # Usar columnas dinámicas para la galería
            n_cols = 3
            cols = st.columns(n_cols)
            
            for i, img in enumerate(imagenes):
                with cols[i % n_cols]:
                    ruta = img.get("ruta_imagen", "")
                    nombre = img.get("nombre_archivo", "imagen")
                    pdf_origen = img.get("pdf_origen", "")
                    pagina = img.get("pagina", 0)
                    # Si la ruta viene de otro PC (absoluta), buscar por nombre en la carpeta local
                    if not ruta or not os.path.exists(ruta):
                        ruta = str(IMAGENES_DIR / nombre)
                    if os.path.exists(ruta):
                        st.image(ruta, caption=f"{pdf_origen} (p.{pagina})", width='stretch')
                    else:
                        st.warning(f"No se encontró: {nombre}")

import urllib.request
import json

def check_api_health():
    """Verifica si la API externa está respondiendo."""
    url = "http://127.0.0.1:8000/health"
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode())
                return f"Conectado"
    except:
        pass
    return "Desconectado"

def obtener_metricas_retrieval():
    """Obtiene las métricas de retrieval desde la API."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/metricas-retrieval", timeout=5) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode())
                return data.get("hit_rate"), data.get("mrr"), data.get("num_preguntas")
    except Exception as e:
        print(f"Error obteniendo métricas de retrieval: {e}")
    return None, None, None

async def ejecutar_streaming(prompt, chat_container):
    full_response = ""
    # Resetear fuentes e imágenes anteriores
    st.session_state.last_sources = []
    st.session_state.last_images = []
    
    response_placeholder = None

    with chat_container:
        # 1. Contenedor para la animación de "Pensando"
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
            <div class="thinking-bubble">
                <span>Analizando documentos</span>
                <div class="thinking-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Crear el placeholder de respuesta SOLO cuando tengamos contenido
        response_placeholder = None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", "http://127.0.0.1:8000/chat/stream", 
                                   json={"pregunta": prompt}) as response:
                
                async for line in response.aiter_lines():
                    if not line: continue
                    
                    if line.startswith("data: "):
                        try:
                            # Cortamos "data: " (6 caracteres) de forma segura
                            json_str = line[6:]
                            token = json.loads(json_str)
                            full_response += token
                        except json.JSONDecodeError as e:
                            print(f"Error decodificando JSON: {e} | Linea: {line}")
                            # Fallback: intentar texto plano si no es JSON (para compatibilidad)
                            token = line.replace("data: ", "")
                            full_response += token
                        
                        # 3. Al recibir el PRIMER TOKEN:
                        # Borramos la animación de "Pensando" y creamos el placeholder de respuesta
                        if response_placeholder is None:
                            thinking_placeholder.empty()  # Limpiar primero
                            with chat_container:
                                response_placeholder = st.empty()  # Crear después
                        
                        # Limpieza de artefactos: eliminar </div> si aparece al final
                        full_response_clean = full_response.replace("</div>", "")

                        # 4. Convertir markdown a HTML progresivamente durante el streaming
                        import markdown
                        content_html = markdown.markdown(full_response_clean)
                        
                        # Renderizar con cursor parpadeante
                        streaming_html = f'''
                        <div class="ai-bubble">
                            <div class="content">
                                {content_html}
                                <span style="animation: blink 1s infinite;">|</span>
                            </div>
                        </div>
                        <style>
                            @keyframes blink {{
                                0%, 50% {{ opacity: 1; }}
                                51%, 100% {{ opacity: 0; }}
                            }}
                        </style>
                        '''
                        response_placeholder.markdown(streaming_html, unsafe_allow_html=True)
                    
                    elif line.startswith("metadata: "):
                        meta_json = json.loads(line.replace("metadata: ", ""))
                        st.session_state.last_sources = meta_json.get("fuentes", [])
                        st.session_state.last_images = meta_json.get("imagenes", [])
                        st.session_state.debug_logs.append(meta_json["debug"])
                        
                        # Actualizar métricas de retrieval después de cada pregunta
                        hit_rate, mrr, num_preguntas = obtener_metricas_retrieval()
                        st.session_state.retrieval_metrics["hit_rate"] = hit_rate
                        st.session_state.retrieval_metrics["mrr"] = mrr
                        st.session_state.retrieval_metrics["num_preguntas"] = num_preguntas
                        
        # 5. Finalización
        thinking_placeholder.empty()
        
        if not full_response:
             full_response = "Lo siento, parece que no tengo información suficiente en este momento para responder a tu pregunta."
             st.session_state.last_sources = []
             st.session_state.last_images = []

        if response_placeholder is not None:
            response_placeholder.empty()
        
        # Limpieza final de artefactos
        full_response = full_response.replace("</div>", "")

        render_message(
            "assistant", 
            full_response, 
            st.session_state.get("last_sources", []),
            st.session_state.get("last_images", [])
        )
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": st.session_state.get("last_sources", []),
            "imagenes": st.session_state.get("last_images", [])
        })

    except Exception as e:
        thinking_placeholder.empty()
        st.error(f"Error de comunicación con la API: {e}")
                    
def main():
    if "api_status" not in st.session_state:
        st.session_state.api_status = check_api_health()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []

    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    # Obtener métricas de retrieval al inicio (solo una vez)
    if "retrieval_metrics" not in st.session_state:
        hit_rate, mrr = obtener_metricas_retrieval()
        st.session_state.retrieval_metrics = {
            "hit_rate": hit_rate,
            "mrr": mrr
        }

    landing_placeholder = st.empty()
    chat_container = st.container()

    if len(st.session_state.messages) == 0 and not st.session_state.processing:
        # Estilo dinámico para posicionar el input y BLOQUEAR SCROLL
        with landing_placeholder.container():
            st.markdown("""
                <style>
                    /* ELIMINAR MÁRGENES GLOBALES EN LANDING */
                    .block-container {
                        padding: 0 !important;
                        max-width: 700px !important;
                        margin: 0 auto !important;
                    }

                    .main {
                        overflow: hidden !important;
                    }

                    /* RESET DE TÍTULOS SOLO EN LANDING */
                    .full-landing-view h1 {
                        margin-top: 0 !important;
                        margin-left: 0 !important;
                    }
                    
                    /* Chat input abajo para no tapar las sugerencias */
                    div[data-testid="stChatInput"] {
                        /* Chat input dentro de columna - sin max-width aquí */
                        margin-top: 4rem !important;
                        background: #FFFFFF !important;
                        border-radius: 0.6rem !important;
                        border: 2px solid #E1CBCB !important;
                        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05) !important;
                    }

                    /* File uploader más compacto y bonito */
                    .stFileUploader > div {
                        margin-top: 4rem !important;
                        padding: 0.5rem !important;
                        border-radius: 0.6rem !important;
                        border: 2px solid #E1CBCB !important;
                        background: #FFFFFF !important;
                        min-height: 60px !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                    }

                    /* Contenedor principal de la landing: padding abajo para que las sugerencias no queden bajo el input */
                    .full-landing-view {
                        display: flex;
                        flex-direction: column;
                        align-items: flex-start;
                        width: 100%;
                        padding-bottom: 140px !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Abrimos el contenedor de la vista completa
            st.markdown('<div class="full-landing-view">', unsafe_allow_html=True)
            
            st.markdown('''
                <div class="landing-title">
                    <h1>Hola,</h1>
                    <h2>¿En qué puedo ayudarte?</h2>
                </div>
            ''', unsafe_allow_html=True)
            
            # Sugerencias
            sugerencias = [
                {"texto": "¿Qué deducciones puedo aplicar como autónomo?"},
                {"texto": "Cómo solicitar subvenciones en Bizkaia"},
                {"texto": "Requisitos para el alta en el RETA"},
                {"texto": "Ayudas para nuevos autónomos en Bizkaia"}
            ]
            
            st.markdown('''<div class="suggestions-title" style="margin-bottom: 0.75rem; font-size: 0.9rem; font-weight: 400;">Puedes empezar por:</div>''', unsafe_allow_html=True)

            # Sugerencias alineadas
            cols = st.columns(2)
            
            for i, sug in enumerate(sugerencias):
                with cols[i % 2]:
                    if st.button(f"{sug['texto']}", key=f"sug_{i}", use_container_width=True):
                        # Al hacer clic, enviamos la sugerencia como pregunta
                        prompt = sug['texto']
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.processing = True
                        st.session_state.awaiting_response = True
                        landing_placeholder.empty() # BORRADO EXPLÍCITO
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True) # Cierre de full-landing-view
    
    else:
        # Modo Chat Normal - FORCE HIDE LANDING VIA CSS Y LIMPIEZA
        landing_placeholder.empty() # ASEGURAR QUE ESTÉ VACÍO
        st.markdown("""
            <style>
                .full-landing-view { display: none !important; }
                div[data-testid="stChatInput"] { 
                    bottom: 40px !important; 
                    transition: bottom 0.3s ease !important;
                }
                .block-container {
                    max-width: 800px !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                }
            </style>
        """, unsafe_allow_html=True)

        with chat_container:
            for msg in st.session_state.messages:
                render_message(
                    msg["role"], 
                    msg["content"], 
                    msg.get("sources"),
                    msg.get("imagenes")
                )
        
        # Si estamos esperando una respuesta, la ejecutamos aquí para que la landing ya esté oculta
        if st.session_state.awaiting_response:
            st.session_state.awaiting_response = False
            last_msg = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
            if last_msg:
                asyncio.run(ejecutar_streaming(last_msg, chat_container))
                st.session_state.processing = False
                st.rerun()

    # Input combinado: texto (75%) + imagen (25%) en una fila - máximo 700px
    st.markdown("""
        <style>
            /* Contenedor centrado con ancho máximo de 700px */
            .input-row-container {
                max-width: 700px;
                margin: 2rem auto 0 auto;
                padding: 0 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Crear columnas con proporciones 75%-25%
    input_cols = st.columns([0.75, 0.25])
    
    with input_cols[0]:
        # Input de texto (75%)
        prompt = st.chat_input("Escribe tu consulta...", disabled=st.session_state.processing, key="main_chat_input")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.processing = True
            st.session_state.awaiting_response = True
            st.rerun()
    
    with input_cols[1]:
        # Input de imagen (25%)
        img_upload = st.file_uploader(
            "",
            type=["png", "jpg", "jpeg"],
            key="buscar_imagen_upload",
            help="Buscar por imagen similar",
            disabled=st.session_state.processing
        )
    
    # Procesar imagen subida (si existe)
    if img_upload is not None:
        # Mostrar preview en un expander
        with st.expander("Vista previa y búsqueda", expanded=True):
            if st.button("Buscar", key="btn_buscar_imagen", use_container_width=True):
                with st.spinner("Buscando..."):
                    try:
                        api_base = "http://127.0.0.1:8000"
                        resp = httpx.post(
                            f"{api_base}/buscar-imagenes",
                            files={"file": (img_upload.name, img_upload.getvalue(), img_upload.type or "image/jpeg")},
                            timeout=30.0,
                        )
                        resp.raise_for_status()
                        resultados = resp.json()
                    except httpx.HTTPStatusError as e:
                        st.error(f"Error de la API: {e.response.status_code}")
                        resultados = []
                    except Exception as e:
                        st.error(f"Error: {e}")
                        resultados = []
                # Añadir como un turno más en el chat
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Buscar imágenes similares a: {img_upload.name}",
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Estas son las **{len(resultados)}** imágenes más similares en la base de datos:" if resultados else "No se encontraron imágenes similares.",
                    "imagenes": resultados,
                })
                st.rerun()

    with st.sidebar:
        st.markdown("<h5>🍁 Asistente Autónomos Bizkaia</h5>", unsafe_allow_html=True)
        st.markdown("<h4>Panel Informativo</h4>", unsafe_allow_html=True)
        
        status_color = f"<span style='color: #2BB92B'>{st.session_state.api_status}</span>" if st.session_state.api_status == "Conectado" else f"<span style='color:red'>{st.session_state.api_status}</span>"
        st.markdown(f"<div style='font-size: 15px; text-align: center;'>API: <b><i>{status_color}</i></b></div>", unsafe_allow_html=True)
        
        st.divider()

        # --- INFORMACIÓN DE DEBUG (CONDICIONAL) ---
        if st.session_state.debug_logs:
            last_log = st.session_state.debug_logs[-1]
            cat = last_log.get("categoria", "N/A")
            
            st.subheader("Análisis del Router")
            st.info(f"Categoría Detectada: {cat}")
            
            st.subheader("Documentos Recuperados")
            sources = st.session_state.get("last_sources", [])
            if not sources:
                st.warning("No se recuperaron documentos de la base de datos.")
            else:
                for s in sources:
                    with st.expander(f"{s['archivo']}"):
                        st.caption(f"Score: {round(s['score'], 3)}")
            
            # Mostrar información de imágenes
            imagenes = st.session_state.get("last_images", [])
            if imagenes:
                st.subheader(f"Imágenes ({len(imagenes)})")
                for img in imagenes:
                    with st.expander(f"{img.get('nombre_archivo', 'imagen')}"):
                        st.caption(f"Origen: {img.get('pdf_origen', 'N/A')}")
                        st.caption(f"Página: {img.get('pagina', 'N/A')}")
                        st.caption(f"Score: {round(img.get('score', 0), 3)}")
                        # Mostrar miniatura si la ruta existe
                        ruta = img.get("ruta_imagen", "")
                        if ruta and os.path.exists(ruta):
                            st.image(ruta, width=150)
            
            st.subheader("Traza del Grafo")
            for i, step in enumerate(last_log.get("pipeline", [])):
                st.caption(f"Paso {i+1}: {step}")
            
            # --- SECCIÓN DE CALIDAD ---
            metricas = last_log.get("metricas", {})
            if metricas:
                st.divider()
                st.subheader("Métricas de Calidad")
                
                c1, c2 = st.columns(2)
                fidelidad = metricas.get("fidelidad")
                relevancia = metricas.get("relevancia")

                with c1:
                    st.markdown("**Fidelidad**")
                    if fidelidad == 1:
                        st.success("✅ Fiel")
                    elif fidelidad == 0:
                        st.error("⚠️ Alucinación")
                    else:
                        st.info("N/A")

                with c2:
                    st.markdown("**Relevancia**")
                    if isinstance(relevancia, (int, float)):
                        if relevancia >= 4:
                            st.success(f"({relevancia}/5)")
                        elif relevancia >= 3:
                            st.warning(f"({relevancia}/5)")
                        else:
                            st.error(f"({relevancia}/5)")
                    else:
                        st.info("N/A")
        else:
            st.info("Realiza una consulta para ver el flujo de datos.")

        st.divider()

        # --- MÉTRICAS DE RETRIEVAL (SIEMPRE VISIBLES) ---
        
        hit_rate = st.session_state.retrieval_metrics.get("hit_rate")
        mrr = st.session_state.retrieval_metrics.get("mrr")
        num_preguntas = st.session_state.retrieval_metrics.get("num_preguntas")

        st.markdown("**Métricas de Retrieval**")
        st.write(f"Número de preguntas (@K): {num_preguntas}")
        col1, col2 = st.columns(2)
        
        with col1:
            if hit_rate is not None:
                try:
                    st.metric("Hit Rate @K", f"{hit_rate*100:.1f}%", border=True)
                except Exception:
                    st.write(f"Hit Rate @K: {hit_rate*100:.1f}%")
            else:
                st.info("Hit Rate: N/A")
        
        with col2:
            if mrr is not None:
                try:
                    st.metric("MRR @K", f"{mrr:.3f}", border=True)
                except Exception:
                    st.write(f"MRR @K: {mrr:.3f}")
            else:
                st.info("MRR: N/A")

if __name__ == "__main__":
    main()