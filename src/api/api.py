import os
import sys
import time
from loguru import logger
from typing import List, Optional, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

from utilidades import utils, funciones_db
from utilidades import prompts

load_dotenv()

# Validacion de variables de entorno
REQUIRED_VARS = [
    "DB_PATH", "COLLECTION_NAME", "MODELO_EMBEDDINGS", 
    "LLM_BASE_URL", "LLM_API_KEY", "MODELO_LLM", "MODELO_FAST"
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

# Carga todas las herramientas necesarias al iniciar la API
@asynccontextmanager
async def lifespan(app: FastAPI):
    # El código aquí se ejecuta al INICIAR la API
    global model_emb, rerank_model, llm_fast, llm_heavy
    dispositivo = "cuda" if os.getenv("USE_CUDA") == "true" else "cpu"
    model_emb = SentenceTransformer(os.getenv("MODELO_EMBEDDINGS"), device=dispositivo)
    rerank_model = CrossEncoder('BAAI/bge-reranker-v2-m3', device=dispositivo)
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
        temperature=0.2
    )

    logger.info("Modelos cargados y listos.")
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
    respuesta_final: str
    debug_pipeline: List[str]
    destino: Optional[str]

def generar_hyde(pregunta, client_llm)->str:
    """
    Alucina una respuesta para mejorar la búsqueda.

    Args:
        pregunta (str): Pregunta del usuario.
        client_llm (ChatOpenAI): Cliente de LLM.

    Returns:
        str: Respuesta alucinada.
    """
    system_prompt = prompts.get_hyde_prompt()
    try:
        r = client_llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": pregunta}])
        return r.content
    except: 
        return pregunta

def nodo_router(state: GraphState):
    """
    Nodo router que decide si es saludo o pregunta médica.
    """
    pregunta = state["pregunta"]
    logger.info(f"[ROUTER] Analizando: {pregunta}")
    llm = llm_fast
    
    system_prompt = prompts.get_router_prompt(CATEGORIAS_VALIDAS)
    
    user_prompt =   f"PREGUNTA DEL USUARIO: '{pregunta}'"

    try:
        clasificacion = llm.invoke([
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]).content.strip().replace("'", "").replace('"', "")
    except Exception as e:
        logger.error(f"Error Router: {e}")
        clasificacion = "otros"

    clasificacion = clasificacion.strip()

    if clasificacion == "SALUDO":
        logger.info("[ROUTER] Detectado SALUDO.")
        state["respuesta_final"] = "¡Hola! Estoy aquí para ayudarte con temas de empleo, ayudas, subvenciones y temas fiscales. Pregúntame lo que quieras."
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

def nodo_buscador(state: GraphState):
    """
    Nodo buscador que aplica filtro por categoría y HyDE.
    """
    cat = state.get("categoria_detectada", "otros")
    pregunta = state["pregunta"]
    
    state["debug_pipeline"].append(f"[BUSCADOR] Filtrando por '{cat}' + HyDE.")
    
    filtro = {"categoria": cat} if cat != "otros" else None
    
    doc_hyde = generar_hyde(pregunta, llm_fast)
    state["debug_pipeline"].append(f"[BUSCADOR] HyDE imaginó: '{doc_hyde[:50]}...'")

    q_emb = utils.generar_embeddings(model_emb, [doc_hyde])
    
    col = funciones_db.obtener_coleccion()
    
    logger.info(f"[BUSCADOR] Buscando documentos por filtro '{filtro}'")
    res = col.query(
        query_embeddings=q_emb,
        n_results=5,     
        where=filtro 
    )
    state["debug_pipeline"].append(f"[BUSCADOR] Encontrados: {len(res['documents'][0])} documentos.")
    logger.info(f"[BUSCADOR] Encontrados: {len(res['documents'][0])} documentos.")
    
    docs = res['documents'][0]
    metas = res['metadatas'][0]
    
    if not docs and filtro:
        state["debug_pipeline"].append("[BUSCADOR] Nada en esa categoría. Buscando en todo...")
        res = col.query(query_embeddings=q_emb, n_results=5)
        docs, metas = res['documents'][0], res['metadatas'][0]

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
    umbral = -0.5  # Umbral más permisivo para no filtrar todos los documentos
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

def nodo_evaluador(state: GraphState):
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
    
    system_prompt = prompts.get_evaluator_prompt()
    
    user_prompt = f"PREGUNTA: {pregunta}\n\nDOCUMENTOS:\n{docs}"
    
    calificacion = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]).content.strip().upper()

    if "SÍ" in calificacion or "SI" in calificacion:
        state["debug_pipeline"].append("[EVALUADOR] Documentos válidos. Procediendo a generar.")
        logger.info("[EVALUADOR] Documentos válidos. Procediendo a generar.")
        state["destino"] = "generador"
    else:
        state["debug_pipeline"].append("[EVALUADOR] Documentos irrelevantes. Abortando generación.")
        logger.info("[EVALUADOR] Documentos irrelevantes. Abortando generación.")
        state["destino"] = "sin_informacion"
    
    return state

def nodo_generador(state: GraphState):
    """
    Nodo generador que redacta la respuesta final.
    """
    state["debug_pipeline"].append("[GENERADOR] Redactando respuesta...")
    logger.info("[GENERADOR] Redactando respuesta...")
    contexto = "\n\n".join(state["contexto_docs"])
    
    if not contexto:
        state["respuesta_final"] = "Lo siento, no tengo información suficiente en mis guías para responder a tu pregunta."
        return state

    system_prompt = prompts.get_generator_prompt()

    user_prompt =   f"""
                    CONTEXTO RECUPERADO ({state.get('categoria_detectada', 'otros')}):
                    {contexto}
                    
                    PREGUNTA DEL USUARIO: "{state["pregunta"]}"
                    
                    IMPORTANTE: Proporciona una respuesta COMPLETA y DETALLADA basada en el contexto. 
                    No te limites a dar el título o una sola frase. Explica la información relevante,
                    incluye pasos si es necesario, y asegúrate de que tu respuesta sea útil y práctica.
                    """

    llm = llm_heavy
    
    respuesta = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1500  # Forzar respuestas más largas
    )
    state["debug_pipeline"].append(f"[GENERADOR] Respuesta generada: {respuesta.content[:100]}")
    logger.info(f"[GENERADOR] Respuesta generada: {respuesta.content[:100]}")
    state["respuesta_final"] = respuesta.content
    return state

def construir_grafo():
    workflow = StateGraph(GraphState)               # Inicializamos el grafo
    workflow.add_node("router", nodo_router)        # Agregamos el nodo router
    workflow.add_node("buscador", nodo_buscador)    # Agregamos el nodo buscador
    workflow.add_node("reranker", nodo_reranker)    # Agregamos el nodo reranker
    workflow.add_node("evaluador", nodo_evaluador)  # Agregamos el nodo evaluador
    workflow.add_node("generador", nodo_generador)  # Agregamos el nodo generador

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

    workflow.add_edge("generador", END)
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

class RespuestaResponse(BaseModel):
    respuesta: str
    fuentes: List[Fuente]
    debug_info: dict
    tiempo_segundos: float

@app.get("/health")
def health(): return {"status": "OK", "version": "1.0"}

@app.post("/chat", response_model=RespuestaResponse)
async def chat_endpoint(request: PreguntaRequest):
    t_start = time.time()
    
    inputs = {
        "pregunta": request.pregunta,
        "historial": request.historial,
        "contexto_docs": [],
        "contexto_fuentes": [],
        "respuesta_final": "",
        "debug_pipeline": [],
        "destino": None,
        "categoria_detectada": "otros"
    }
    res = await app_graph.ainvoke(inputs)
    return RespuestaResponse(
        respuesta=res["respuesta_final"],
        fuentes=[Fuente(**f) for f in res["contexto_fuentes"]],
        debug_info={
            "pipeline": res["debug_pipeline"], 
            "categoria": res.get("categoria_detectada")
        },
        tiempo_segundos=time.time() - t_start
    )

@app.post("/chat/stream")
async def chat_streaming_endpoint(request: PreguntaRequest):
    inputs = {
        "pregunta": request.pregunta,
        "historial": request.historial,
        "contexto_docs": [],
        "contexto_fuentes": [],
        "respuesta_final": "",
        "debug_pipeline": [],
        "destino": None,
        "categoria_detectada": "otros"
    }
    async def generate():
        last_state = {}
        content_yielded = False
        
        async for event in app_graph.astream_events(inputs, version="v2"):
            # 1. Capturamos tokens del modelo
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {content}\n\n" # Formato Server-Sent Events simplificado
                    content_yielded = True
            
            # 2. Capturamos el estado final cuando el grafo termina
            elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                last_state = event["data"]["output"]
        
        # 2b. Fallback: Si no hubo streaming (ej: Saludo estático), enviamos la respuesta final de golpe
        if not content_yielded and last_state and last_state.get("respuesta_final"):
            yield f"data: {last_state.get('respuesta_final')}\n\n"

        # 3. Enviamos los metadatos al final como un último mensaje
        if last_state:
            metadata = {
                "fuentes": last_state.get("contexto_fuentes", []),
                "debug": {
                    "categoria": last_state.get("categoria_detectada"),
                    "pipeline": last_state.get("debug_pipeline")
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