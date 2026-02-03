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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente RAG Bizkaia",
    page_icon="",
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
        background-color: #FCF8F8; /* Slate 50 */
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(246 59 59 / 0.05) 0%, transparent 50%);
        background-attachment: fixed;
    }

    h1 {
        background: linear-gradient(135deg, #FF0000 0%, #FC7171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 250 !important;
        font-size: 2.5rem !important;
        margin-bottom: -1.5rem !important;
        letter-spacing: -0.5px;
    }
    h2 {
        background: linear-gradient(135deg, #FC7171 0%, #FFD5D5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 500 !important;
        font-size: 3rem !important;
        margin-top: -1.5rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.5px;
    }
    h4 {
        background: linear-gradient(135deg, #FC7171 0%, #FFD5D5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 350 !important;
        font-size: 2rem !important;
        letter-spacing: -0.5px;
        text-align: center !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
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
        background-color: #FFFFFF;
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
        background-color: #FFFFFF;
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
    
    /* INPUT FLOTANTE (ADAPTIVE) */
    .stChatInput {
        /* Eliminamos posicionamiento forzado para que respete el sidebar nativamente */
        bottom: 40px !important;
        background: transparent !important;
    }
    
    div[data-testid="stChatInput"] {
        /* Estilo Isla centrado en su contenedor */
        max-width: 800px !important; 
        margin: 0 auto !important;
        
        background: #FFFFFF !important;
        border-radius: 0.6rem !important;
        border: 2px solid #E1CBCB !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05) !important;
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
        background-color: #FFFFFF;
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
    
    .image-caption {
        padding: 0.5rem;
        font-size: 0.75rem;
        color: #694747;
        background: #FFF8F8;
    }
</style>
"""
st.markdown(ST_STYLE, unsafe_allow_html=True)

def render_message(role, content, sources=None, imagenes=None):
    """Renderiza un mensaje con estilos HTML personalizados."""
    if not content or not content.strip():
        return

    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        # Convertir markdown a HTML para evitar problemas de anidación
        import markdown
        content_html = markdown.markdown(content)
        
        sources_html = ""
        if sources:
            sources_html = "".join([
                f'<a href="#" class="source-chip">{s.get("archivo", "Doc")} ({s.get("chunk_id", "?")})</a>' 
                for s in sources
            ])
            sources_html = f'<div class="sources-container">{sources_html}</div>'
        
        # Galería de imágenes
        images_html = ""
        if imagenes and len(imagenes) > 0:
            images_cards = ""
            for img in imagenes:
                ruta = img.get("ruta_imagen", "")
                nombre = img.get("nombre_archivo", "imagen")
                pdf_origen = img.get("pdf_origen", "")
                pagina = img.get("pagina", 0)
                
                # Usar st.image después del HTML para mejor compatibilidad
                # Primero mostramos el HTML y luego usamos columnas de Streamlit
                images_cards += f'''
                <div class="image-card">
                    <img src="{ruta}" alt="{nombre}" />
                    <div class="image-caption">
                        📄 {pdf_origen} (p.{pagina})
                    </div>
                </div>
                '''
            
            images_html = f'''
            <div class="image-gallery">
                <div class="image-gallery-title">🖼️ Imágenes relacionadas ({len(imagenes)})</div>
                <div class="image-grid">
                    {images_cards}
                </div>
            </div>
            '''
        
        full_html = f'''
        <div class="ai-bubble">
            <div class="content">
                {content_html}
            </div>
            {sources_html}
            {images_html}
        </div>
        '''
        st.markdown(full_html, unsafe_allow_html=True)

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

async def ejecutar_streaming(prompt, chat_container):
    full_response = ""
    # Resetear fuentes e imágenes anteriores
    st.session_state.last_sources = []
    st.session_state.last_images = []
    
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
                        token = line.replace("data: ", "")
                        full_response += token
                        
                        # 3. Al recibir el PRIMER TOKEN:
                        # Borramos la animación de "Pensando" y creamos el placeholder de respuesta
                        if response_placeholder is None:
                            thinking_placeholder.empty()  # Limpiar primero
                            with chat_container:
                                response_placeholder = st.empty()  # Crear después
                        
                        # 4. Convertir markdown a HTML progresivamente durante el streaming
                        import markdown
                        content_html = markdown.markdown(full_response)
                        
                        # Renderizar con cursor parpadeante
                        streaming_html = f'''
                        <div class="ai-bubble">
                            <div class="content">
                                {content_html}
                                <span style="animation: blink 1s infinite;">▌</span>
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
                        
        # 5. Finalización: Limpiar placeholders y renderizar mensaje final estático
        thinking_placeholder.empty()
        if response_placeholder is not None:
            response_placeholder.empty()
        
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
    st.markdown("<h1>&nbsp;&nbsp;&nbsp;Hola,</h1>", unsafe_allow_html=True)
    st.markdown("<h2>¿En qué puedo ayudarte?</h2>", unsafe_allow_html=True)
    
    if "api_status" not in st.session_state:
        st.session_state.api_status = check_api_health()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []

    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_message(
                msg["role"], 
                msg["content"], 
                msg.get("sources"),
                msg.get("imagenes")
            )

    if prompt := st.chat_input("Escribe tu consulta..."):
        with chat_container:
            render_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Llamamos a la lógica de streaming asíncrona
        asyncio.run(ejecutar_streaming(prompt, chat_container))

    with st.sidebar:
        st.markdown("<h4>Panel de Control</h4>", unsafe_allow_html=True)
        
        status_color = f"<span style='color:lime'>{st.session_state.api_status}</span>" if st.session_state.api_status == "Conectado" else f"<span style='color:red'>{st.session_state.api_status}</span>"
        st.markdown(f"<h5 style='text-align: center;'>API: {status_color}</h5>", unsafe_allow_html=True)
        
        if st.session_state.debug_logs:
            last_log = st.session_state.debug_logs[-1]
            cat = last_log.get("categoria", "N/A")
            
            st.divider()
            st.subheader("Análisis del Router")
            st.info(f"Categoría Detectada: {cat}")
            
            st.subheader("Documentos Recuperados")
            sources = st.session_state.get("last_sources", [])
            if not sources:
                st.warning("No se recuperaron documentos de la base de datos.")
            else:
                for s in sources:
                    with st.expander(f"{s['archivo']}"):
                        st.caption(f"ID: {s['chunk_id']} | Score: {round(s['score'], 3)}")
            
            # Mostrar información de imágenes
            imagenes = st.session_state.get("last_images", [])
            if imagenes:
                st.subheader(f"🖼️ Imágenes ({len(imagenes)})")
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
                st.caption(f"Step {i+1}: {step}")
        else:
            st.info("Realiza una consulta para ver el flujo de datos.")

if __name__ == "__main__":
    main()