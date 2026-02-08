import io
import os
import sys
import time
from loguru import logger
from typing import List, Optional, TypedDict
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import uvicorn
from pathlib import Path
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
import json

# Agregar src al path para asegurar imports si se ejecuta directamente
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
load_dotenv()

from utilidades import utils, funciones_db, funciones_evaluacion
from utilidades import prompts
import torch
from transformers import CLIPModel, CLIPProcessor

# Definir la ruta del golden set usando la variable de entorno si está disponible
GOLDEN_SET_FILE = os.getenv(
    "GOLDEN_SET_FILE",
    str(utils.project_root() / "src" / "golden_set_automatico.jsonl")
)

# Validacion de variables de entorno
REQUIRED_VARS = [
    "DB_PATH", "COLLECTION_NAME_PDFS", "COLLECTION_NAME_IMAGENES", "MODELO_EMBEDDINGS", 
    "LLM_BASE_URL", "LLM_API_KEY", "MODELO_LLM", "MODELO_FAST", "MODELO_CLIP", "MODELO_RERANKER"
]

missing_vars = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing_vars:
    logger.critical(f"Faltan variables de entorno críticas: {missing_vars}")
    raise RuntimeError(f"Configuración incompleta. Faltan: {missing_vars}")

CATEGORIAS_VALIDAS = ["Laboral", "Fiscal", "Ayudas_y_Subvenciones"]

model_emb       = None
rerank_model    = None
llm_fast        = None
llm_heavy       = None
model_clip      = None
clip_processor  = None
device          = None

# Caché de métricas de retrieval (se rellenan al arranque o al llamar a /metricas-retrieval)
retrieval_metrics_cache: dict = {"hit_rate": None, "mrr": None, "num_preguntas": None}


def _evaluar_retrieval_y_guardar_en_cache() -> None:
    """Ejecuta la evaluación de retrieval una vez y guarda hit_rate y mrr en caché."""
    global model_emb
    try:
        col_pdfs = funciones_db.obtener_coleccion("pdfs")
        golden_file = os.getenv("GOLDEN_SET_FILE", GOLDEN_SET_FILE)
        golden_set = []
        if os.path.exists(golden_file):
            with open(golden_file, "r", encoding="utf-8") as f:
                golden_set = [json.loads(line) for line in f if line.strip()]
            logger.info(f"[RETRIEVAL] Golden set cargado ({len(golden_set)} entradas) desde {golden_file}.")
        else:
            num = int(os.getenv("GOLDEN_SET_DEFAULT_NUM", "20"))
            golden_set = funciones_evaluacion.crear_golden_set_automatico(
                col_pdfs, llm_fast, os.getenv("MODELO_FAST"), num_preguntas=num
            )
        if golden_set:
            hit_rate, mrr = funciones_evaluacion.evaluar_retrieval(
                col_pdfs, model_emb, golden_set, top_k=3
            )
            retrieval_metrics_cache["hit_rate"] = hit_rate
            retrieval_metrics_cache["mrr"] = mrr
            retrieval_metrics_cache["num_preguntas"] = num
            logger.info(f"[RETRIEVAL] Métricas en caché: hit_rate={hit_rate:.2%}, mrr={mrr:.3f}, num_preguntas={num}")
        else:
            logger.warning("[RETRIEVAL] No hay golden set; métricas de retrieval quedarán en None.")
    except Exception as e:
        logger.error(f"[RETRIEVAL] Error evaluando retrieval al inicio: {e}")


# Carga todas las herramientas necesarias al iniciar la API
@asynccontextmanager
async def lifespan(app: FastAPI):
    # El código aquí se ejecuta al INICIAR la API
    global model_emb, rerank_model, llm_fast, llm_heavy, model_clip, clip_processor, device
    device = "cuda" if os.getenv("USE_CUDA") == "true" else "cpu"
    
    logger.info(f"Cargando modelos en dispositivo: {device.upper()}")
    
    model_emb = SentenceTransformer(os.getenv("MODELO_EMBEDDINGS"), device=device)
    rerank_model = CrossEncoder(os.getenv("MODELO_RERANKER", 'BAAI/bge-reranker-v2-m3'), device=device)
    
    # Cargar CLIP
    logger.info("Cargando modelo CLIP...")
    model_clip = CLIPModel.from_pretrained(os.getenv("MODELO_CLIP")).to(device)
    clip_processor = CLIPProcessor.from_pretrained(os.getenv("MODELO_CLIP"))
    
    llm_fast = ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"), 
        api_key=os.getenv("LLM_API_KEY"), 
        model=os.getenv("MODELO_FAST"), 
        temperature=0
    )

    llm_heavy = ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"), 
        api_key=os.getenv("LLM_API_KEY"), 
        model=os.getenv("MODELO_LLM"), 
        temperature=0.2,
        streaming=True
    )

    logger.info("Modelos cargados y listos.")
    # Evaluar retrieval una sola vez al arranque (evita hacerlo en cada chat)
    if os.getenv("EVALUAR_RETRIEVAL_AL_INICIO", "true").lower() == "true":
        logger.info("Evaluando retrieval (golden set) al inicio...")
        _evaluar_retrieval_y_guardar_en_cache()
    else:
        logger.info("Evaluación de retrieval al inicio desactivada (EVALUAR_RETRIEVAL_AL_INICIO=false).")
    logger.info("Iniciando servicios de RAG...")
    yield
    # El código aquí se ejecuta al CERRAR la API
    logger.info("Cerrando servicios y liberando memoria...")


class GraphState(TypedDict):
    """
    Clase que representa el estado del grafo.
    """
    pregunta: str
    historial: List[dict]
    categoria_detectada: str
    contexto_docs: List[str]
    contexto_fuentes: List[dict]
    imagenes_relacionadas: List[dict]  # Nuevo campo para imágenes
    respuesta_final: str
    metricas: dict
    debug_pipeline: List[str]
    destino: Optional[str]

async def generar_hyde(pregunta, client_llm)->str:
    """
    Alucina una respuesta para mejorar la búsqueda.

    Args:
        pregunta (str): Pregunta del usuario.
        client_llm (ChatOpenAI): Cliente de LLM.

    Returns:
        str: Respuesta alucinada.
    """
    system_prompt = prompts.obtener_prompt_hyde()
    try:
        r = await client_llm.ainvoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": pregunta}])
        return r.content
    except: 
        return pregunta

async def nodo_router(state: GraphState):
    """
    Nodo router que decide si es saludo o pregunta.
    """
    pregunta = state["pregunta"]
    logger.info(f"[ROUTER] Analizando: {pregunta}")
    llm = llm_fast
    
    system_prompt = prompts.obtener_prompt_router(CATEGORIAS_VALIDAS)
    
    user_prompt =   f"PREGUNTA DEL USUARIO: '{pregunta}'"

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ])
        clasificacion = response.content.strip().replace("'", "").replace('"', "")
    except Exception as e:
        logger.error(f"Error Router: {e}")
        clasificacion = "otros"

    clasificacion = clasificacion.strip()

    if clasificacion == "SALUDO":
        logger.info("[ROUTER] Detectado SALUDO.")
        state["respuesta_final"] = "¡Hola! Soy un asistente virtual para autónomos en Bizkaia. Estoy aquí para ayudarte con temas de empleo, ayudas, subvenciones y temas fiscales. Pregúntame lo que quieras."
        state["debug_pipeline"].append("[ROUTER] Detectado SALUDO.")
        state["destino"] = "fin"
        return state

    cat_final = "otros"
    logger.info(f"[ROUTER] Clasificado como '{clasificacion}'.")
    for cat in CATEGORIAS_VALIDAS:
        if cat.lower() in clasificacion.lower():
            cat_final = cat
            break
            
    state["categoria_detectada"] = cat_final
    state["debug_pipeline"].append(f"[ROUTER] Clasificado como '{cat_final}'. Enviando a Buscador...")
    state["destino"] = "buscador"
    return state

async def nodo_buscador(state: GraphState):
    """
    Nodo buscador que aplica filtro por categoría y HyDE.
    Ahora busca en AMBAS colecciones: texto (PDFs) e imágenes.
    """
    cat = state.get("categoria_detectada", "otros")
    pregunta = state["pregunta"]
    
    state["debug_pipeline"].append(f"[BUSCADOR] Filtrando por '{cat}' + HyDE.")
    
    filtro = {"categoria": cat} if cat != "otros" else None
    
    doc_hyde = await generar_hyde(pregunta, llm_fast)
    state["debug_pipeline"].append(f"[BUSCADOR] HyDE imaginó: '{doc_hyde[:50]}...'")

    q_emb = utils.generar_embeddings(model_emb, [doc_hyde])
    
    # ========== BÚSQUEDA EN COLECCIÓN DE PDFs (TEXTO) ==========
    col_pdfs = funciones_db.obtener_coleccion("pdfs")
    
    logger.info(f"[BUSCADOR] Buscando documentos de texto por filtro '{filtro}'")
    res_pdfs = col_pdfs.query(
        query_embeddings=q_emb,
        n_results=5,     
        where=filtro 
    )
    state["debug_pipeline"].append(f"[BUSCADOR] Encontrados: {len(res_pdfs['documents'][0])} documentos de texto.")
    logger.info(f"[BUSCADOR] Encontrados: {len(res_pdfs['documents'][0])} documentos de texto.")
    
    docs = res_pdfs['documents'][0]
    metas = res_pdfs['metadatas'][0]
    
    if not docs and filtro:
        state["debug_pipeline"].append("[BUSCADOR] Nada en esa categoría. Buscando en todo...")
        res_pdfs = col_pdfs.query(query_embeddings=q_emb, n_results=5)
        docs, metas = res_pdfs['documents'][0], res_pdfs['metadatas'][0]

    state["contexto_docs"] = []
    fuentes = []
    textos_vistos = set() 
    
    for doc, meta in zip(docs, metas):
        texto_final = meta.get("contexto_expandido")
        if texto_final:
            state["debug_pipeline"].append(f"[AUTO-MERGING] Expandido a contexto padre ({len(texto_final)} chars).")
        else:
            texto_final = doc

        if texto_final not in textos_vistos:
            state["contexto_docs"].append(texto_final)
            textos_vistos.add(texto_final)
            
            fuentes.append({
                "archivo": meta.get("source", "desc"),
                "chunk_id": str(meta.get("chunk_index", 0)),
                "score": 0.0, 
                "relevante": True
            })

    state["contexto_fuentes"] = fuentes
    
    # ========== BÚSQUEDA EN COLECCIÓN DE IMÁGENES (CLIP) ==========
    try:
        col_imagenes = funciones_db.obtener_coleccion("imagenes")
        logger.info(f"[BUSCADOR] Buscando imágenes por filtro '{filtro}' Uusando CLIP")
        
        # Generar embedding de texto con CLIP para la pregunta
        # Usamos la pregunta original o una versión corta, ya que CLIP prefiere textos cortos
        texto_query_clip = pregunta[:77] # CLIP suele tener limite de contexto de 77 tokens
        
        inputs = clip_processor(text=texto_query_clip, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model_clip.get_text_features(**inputs)
            
            # Robustez: verificar si devuelve un objeto en lugar de tensor
            if not isinstance(text_features, torch.Tensor):
                if hasattr(text_features, "text_embeds"):
                    text_features = text_features.text_embeds
                elif hasattr(text_features, "pooler_output"):
                    # Fallback: Extraer pooler_output y proyectar si es necesario
                    # Esto maneja el caso donde get_text_features devuelve el output raw del text_model
                    text_features = text_features.pooler_output
                    if hasattr(model_clip, "text_projection"):
                        text_features = model_clip.text_projection(text_features)
            
            # Normalizar es importante para CLIP si usamos distancia coseno
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            q_emb_clip = text_features.cpu().numpy().tolist()

        res_imagenes = col_imagenes.query(
            query_embeddings=q_emb_clip,
            n_results=3,
            where=filtro
        )
        
        imagenes = []
        if res_imagenes['metadatas'][0]:
            imagenes_dir = utils.project_root() / "data" / "documentos" / "imagenes"
            for i, meta in enumerate(res_imagenes['metadatas'][0]):
                score = res_imagenes.get('distances', [[0]*len(res_imagenes['metadatas'][0])])[0][i]
                nombre_archivo = meta.get("nombre_archivo", "")
                # Usar siempre ruta local (nombre_archivo) para que funcione en cualquier equipo
                ruta_local = str(imagenes_dir / nombre_archivo) if nombre_archivo else ""
                imagenes.append({
                    "ruta_imagen": ruta_local,
                    "nombre_archivo": nombre_archivo,
                    "pdf_origen": meta.get("pdf_origen", ""),
                    "pagina": meta.get("pagina", 0),
                    "score": float(score) if score else 0.0
                })
            
            state["debug_pipeline"].append(f"[BUSCADOR] Encontradas {len(imagenes)} imágenes relacionadas.")
            logger.info(f"[BUSCADOR] Encontradas {len(imagenes)} imágenes relacionadas.")
        else:
            state["debug_pipeline"].append("[BUSCADOR] No se encontraron imágenes relacionadas.")
        
        state["imagenes_relacionadas"] = imagenes
        
    except Exception as e:
        logger.warning(f"[BUSCADOR] Error buscando imágenes: {e}")
        state["debug_pipeline"].append(f"[BUSCADOR] Error buscando imágenes: {e}")
        state["imagenes_relacionadas"] = []
    
    return state

def nodo_reranker(state: GraphState):
    """
    Re-ordena los documentos recuperados para asegurar que los mejores 
    estén al principio y descarta los que tienen baja puntuación.
    """
    pregunta = state["pregunta"]
    docs = state["contexto_docs"]
    metas = state["contexto_fuentes"]
    
    if not docs:
        return state

    state["debug_pipeline"].append("[RE-RANKER] Evaluando relevancia semántica...")
    logger.info(f"[RE-RANKER] Evaluando relevancia semántica...")

    # Preparamos los pares (pregunta, documento) para el modelo
    pairs = [[pregunta, doc] for doc in docs]
    
    # El modelo devuelve una puntuación para cada par
    scores = rerank_model.predict(pairs)
    
    # Combinamos, ordenamos por score y filtramos
    scored_docs = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
    
    # Filtro de corte: Solo nos quedamos con los que superen un umbral (ej: 0.1 o 0.3)
    # y limitamos a los 3 mejores para no saturar el contexto del LLM
    umbral = 0.25
    docs_reordenados = []
    metas_reordenadas = []
    
    for score, doc, meta in scored_docs:
        if score >= umbral:
            meta["score"] = float(score) # Actualizamos el score real
            docs_reordenados.append(doc)
            metas_reordenadas.append(meta)
            
    state["contexto_docs"] = docs_reordenados[:3]
    state["contexto_fuentes"] = metas_reordenadas[:3]
    
    state["debug_pipeline"].append(f"[RE-RANKER] De {len(docs)} docs, mantengo {len(docs_reordenados)} tras re-ranking.")
    logger.info(f"[RE-RANKER] De {len(docs)} docs, mantengo {len(docs_reordenados)} tras re-ranking.")
    
    return state

async def nodo_evaluador(state: GraphState):
    """
    Evalúa si los documentos recuperados son relevantes para la pregunta.
    """
    pregunta = state["pregunta"]
    docs = state["contexto_docs"]
    
    state["debug_pipeline"].append("[EVALUADOR] Calificando relevancia de documentos...")
    logger.info(f"[EVALUADOR] Calificando relevancia de documentos...")
    
    if not docs:
        state["destino"] = "sin_informacion"
        return state

    llm = llm_fast
    
    system_prompt = prompts.obtener_prompt_evaluador()
    
    user_prompt = f"PREGUNTA: {pregunta}\n\nDOCUMENTOS:\n{docs}"
    
    try:
        calificacion = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        calificacion = calificacion.content.strip().upper()
    except Exception as e:
        logger.error(f"Error Evaluador: {e}")
        calificacion = "NO"

    if "SÍ" in calificacion or "SI" in calificacion:
        state["debug_pipeline"].append("[EVALUADOR] Documentos válidos. Procediendo a generar.")
        logger.info("[EVALUADOR] Documentos válidos. Procediendo a generar.")
        state["destino"] = "generador"
    else:
        state["debug_pipeline"].append("[EVALUADOR] Documentos irrelevantes. Abortando generación.")
        logger.info("[EVALUADOR] Documentos irrelevantes. Abortando generación.")
        # FIX: Establecer respuesta final para que no llegue vacía al cliente
        state["respuesta_final"] = "Lo siento, parece que no tengo información suficiente en este momento para responder a tu pregunta. "
        state["destino"] = "sin_informacion"
    
    return state

async def nodo_generador(state: GraphState):
    """
    Nodo generador que redacta la respuesta final.
    """
    state["debug_pipeline"].append("[GENERADOR] Redactando respuesta...")
    logger.info("[GENERADOR] Redactando respuesta...")
    contexto = "\n\n".join(state["contexto_docs"])
    
    if not contexto:
        state["respuesta_final"] = "Lo siento, no tengo información suficiente en mis guías para responder a tu pregunta."
        return state

    system_prompt = prompts.obtener_prompt_generador()

    user_prompt =   f"""
                    CONTEXTO RECUPERADO ({state.get('categoria_detectada', 'otros')}):
                    {contexto}
                    
                    PREGUNTA DEL USUARIO: "{state["pregunta"]}"
                    
                    IMPORTANTE: Proporciona una respuesta COMPLETA y DETALLADA basada en el contexto. 
                    No te limites a dar el título o una sola frase. Explica la información relevante,
                    incluye pasos si es necesario, y asegúrate de que tu respuesta sea útil y práctica.
                    """

    # Configurar el LLM con max_tokens usando bind() para que se aplique también en streaming
    llm_configured = llm_heavy.bind(max_tokens=2000)
    
    respuesta = await llm_configured.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    state["debug_pipeline"].append(f"[GENERADOR] Respuesta generada: {respuesta.content[:100]}")
    logger.info(f"[GENERADOR] Respuesta generada: {respuesta.content[:100]}")
    state["respuesta_final"] = respuesta.content
    return state

async def nodo_calidad(state: GraphState):
    """
    Evalúa la calidad de la respuesta generada (Fidelidad y Relevancia).
    """
    state["debug_pipeline"].append("[CALIDAD] Evaluando respuesta generada...")
    logger.info("[CALIDAD] Evaluando respuesta generada...")
    
    pregunta = state["pregunta"]
    respuesta = state["respuesta_final"]
    contexto = "\n\n".join(state["contexto_docs"])
    
    # Evaluar Fidelidad (¿Alucinaciones?)
    # Usamos llm_fast para rapidez
    fidelidad = funciones_evaluacion.evaluar_fidelidad(
        pregunta, respuesta, contexto, 
        llm_fast, os.getenv("MODELO_FAST")
    )
    
    # Evaluar Relevancia (¿Responde al user?)
    relevancia = funciones_evaluacion.evaluar_relevancia(
        pregunta, respuesta, 
        llm_fast, os.getenv("MODELO_FAST")
    )
    
    # Usar métricas de retrieval en caché (calculadas al arranque de la API)
    hit_rate = retrieval_metrics_cache.get("hit_rate")
    mrr = retrieval_metrics_cache.get("mrr")
    num_preguntas = retrieval_metrics_cache.get("num_preguntas")
    
    state["metricas"] = {
        "fidelidad": fidelidad,
        "relevancia": relevancia,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "num_preguntas": num_preguntas
    }
    state["debug_pipeline"].append(f"[CALIDAD] Fidelidad: {fidelidad} | Relevancia: {relevancia}")
    logger.info(f"[CALIDAD] Fidelidad: {fidelidad} | Relevancia: {relevancia}")
    
    return state

def construir_grafo():
    workflow = StateGraph(GraphState)               # Inicializamos el grafo
    workflow.add_node("router", nodo_router)        # Agregamos el nodo router
    workflow.add_node("buscador", nodo_buscador)    # Agregamos el nodo buscador
    workflow.add_node("reranker", nodo_reranker)    # Agregamos el nodo reranker
    workflow.add_node("evaluador", nodo_evaluador)  # Agregamos el nodo evaluador
    workflow.add_node("generador", nodo_generador)  # Agregamos el nodo generador
    workflow.add_node("calidad", nodo_calidad)      # Agregamos el nodo calidad

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["destino"],
        {
            "fin": END,
            "buscador": "buscador"
        }
    )
    workflow.add_edge("buscador", "reranker")
    workflow.add_edge("reranker", "evaluador")
    workflow.add_conditional_edges(
        "evaluador",
        lambda state: state["destino"],
        {
            "generador": "generador",
            "sin_informacion": END
        }
    )

    workflow.add_edge("generador", "calidad")
    workflow.add_edge("calidad", END)
    return workflow.compile()

app_graph = construir_grafo()
app = FastAPI(title="API RAG Autonomos Bizkaia", version="1.0", lifespan=lifespan)

class PreguntaRequest(BaseModel):
    pregunta: str
    historial: Optional[List[dict]] = []

class Fuente(BaseModel):
    archivo: str
    chunk_id: str
    score: float
    relevante: bool

class Imagen(BaseModel):
    ruta_imagen: str
    nombre_archivo: str
    pdf_origen: str
    pagina: int
    score: float

class RespuestaResponse(BaseModel):
    respuesta: str
    fuentes: List[Fuente]
    imagenes: List[Imagen]
    debug_info: dict
    tiempo_segundos: float

@app.get("/health")
def health(): return {"status": "OK", "version": "1.0"}


@app.get("/metricas-retrieval")
def get_metricas_retrieval():
    """Devuelve las métricas de retrieval en caché (calculadas al arranque)."""
    return {
        "hit_rate": retrieval_metrics_cache.get("hit_rate"),
        "mrr": retrieval_metrics_cache.get("mrr"),
        "num_preguntas": retrieval_metrics_cache.get("num_preguntas")
    }


@app.post("/metricas-retrieval")
def refrescar_metricas_retrieval():
    """Vuelve a evaluar el retrieval y actualiza la caché (útil tras añadir documentos)."""
    _evaluar_retrieval_y_guardar_en_cache()
    return {
        "hit_rate": retrieval_metrics_cache.get("hit_rate"),
        "mrr": retrieval_metrics_cache.get("mrr"),
        "num_preguntas": retrieval_metrics_cache.get("num_preguntas")
    }


@app.post("/buscar-imagenes", response_model=List[Imagen])
async def buscar_imagenes_similares(file: UploadFile = File(...)):
    """
    Sube una imagen y devuelve las imágenes de la base de datos más similares (CLIP).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (jpg, png, etc.).")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.warning(f"Error leyendo imagen: {e}")
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen.")
    try:
        col_imagenes = funciones_db.obtener_coleccion("imagenes")
    except Exception as e:
        logger.warning(f"Error obteniendo colección imágenes: {e}")
        raise HTTPException(status_code=503, detail="Base de datos de imágenes no disponible.")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model_clip.get_image_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif hasattr(features, "image_embeds"):
            features = features.image_embeds
        if not isinstance(features, torch.Tensor):
            features = features[0] if isinstance(features, (list, tuple)) else features
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        q_emb = features.cpu().numpy().tolist()
    res = col_imagenes.query(query_embeddings=q_emb, n_results=3)
    imagenes_dir = utils.project_root() / "data" / "documentos" / "imagenes"
    out = []
    if res["metadatas"] and res["metadatas"][0]:
        for i, meta in enumerate(res["metadatas"][0]):
            nombre_archivo = meta.get("nombre_archivo", "")
            score = res.get("distances", [[0] * len(res["metadatas"][0])])[0][i]
            ruta_local = str(imagenes_dir / nombre_archivo) if nombre_archivo else ""
            out.append(Imagen(
                ruta_imagen=ruta_local,
                nombre_archivo=nombre_archivo,
                pdf_origen=meta.get("pdf_origen", ""),
                pagina=meta.get("pagina", 0),
                score=float(score) if score is not None else 0.0,
            ))
    return out

@app.post("/chat/stream")
async def chat_streaming_endpoint(request: PreguntaRequest):
    inputs = {
        "pregunta": request.pregunta,
        "historial": request.historial,
        "contexto_docs": [],
        "contexto_fuentes": [],
        "imagenes_relacionadas": [],
        "respuesta_final": "",
        "debug_pipeline": [],
        "destino": None,
        "categoria_detectada": "otros",
        "metricas": {}
    }
    async def generate():
        last_state = {}
        content_yielded = False
        
        async for event in app_graph.astream_events(inputs, version="v2"):
            # 1. Capturamos tokens del modelo
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps(content)}\n\n" # Usar JSON para escapar newlines
                    content_yielded = True
            
            # 2. Capturamos el estado final cuando el grafo termina
            elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                last_state = event["data"]["output"]
        
        # 2b. Fallback: Si no hubo streaming (ej: Saludo estático), enviamos la respuesta final de golpe
        if not content_yielded and last_state and last_state.get("respuesta_final"):
            yield f"data: {json.dumps(last_state.get('respuesta_final'))}\n\n"

        # 3. Enviamos los metadatos al final como un último mensaje
        if last_state:
            metadata = {
                "fuentes": last_state.get("contexto_fuentes", []),
                "imagenes": last_state.get("imagenes_relacionadas", []),
                "debug": {
                    "categoria": last_state.get("categoria_detectada"),
                    "pipeline": last_state.get("debug_pipeline"),
                    "metricas": last_state.get("metricas", {})
                }
            }
            yield f"metadata: {json.dumps(metadata)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

def ejecutar_api():
    """Ejecuta la API de RAG para Autónomos Bizkaia"""
    host = "127.0.0.1" 
    port = 8000
    
    try:
        logger.info(f"Iniciando servidor en http://{host}:{port}")
        uvicorn.run(
            "api.api:app", 
            host=host, 
            port=port,
            reload=True,               
            workers=1                  
        )
    except Exception as e:
        logger.error(f"Error al iniciar la API: {e}")