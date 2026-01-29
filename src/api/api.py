"""
API para la aplicacion de RAG
"""

import os
import sys
import time
from loguru import logger
from typing import List, Optional, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

DB_DIR = os.getenv("DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
MODELO_LLM = os.getenv("MODELO_LLM")
MODELO_FAST = os.getenv("MODELO_FAST")

CATEGORIAS_VALIDAS = ["Laboral", "Fiscal", "Ayudas_y_Subvenciones"]

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
    system_prompt = """
                    Actúa como un Asesor Técnico de la Hacienda Foral de Bizkaia y experto en Gestión Administrativa.
                    Tu tarea es convertir la consulta del usuario en un fragmento de manual técnico o normativa foral (estilo extracto de Bizkaia.eus o Reglamento del Impuesto sobre Actividades Económicas).

                    Instrucciones:
                    - No respondas al usuario directamente.
                    - Traduce el lenguaje coloquial a terminología administrativa y tributaria precisa (ej: usa 'Hecho Imponible', 'Domicilio Fiscal', 'Modelo 036', 'TicketBai/Batuz', 'Exención', 'Censo de Entidades').
                    - Redacta un párrafo breve, formal y descriptivo que contenga la información teórica necesaria para resolver la consulta.
                    - Asegúrate de mencionar conceptos específicos de la normativa de Bizkaia si son relevantes.
                    """
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
    llm = ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=MODELO_FAST, temperature=0)
    
    system_prompt = f"""
                    Eres un Clasificador de Intenciones experto.
                    Tu trabajo es categorizar la pregunta del usuario en una de las siguientes opciones:
                    1. SALUDO (charla trivial).
                    2. CATEGORIAS: {', '.join(CATEGORIAS_VALIDAS)}.
                    
                    Responde ÚNICAMENTE con la palabra de la categoría (o 'SALUDO'). Nada más.
                    """
    
    user_prompt =   f"PREGUNTA DEL USUARIO: '{pregunta}'"

    try:
        clasificacion = llm.invoke([
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]).content.strip().replace("'", "").replace('"', "")
    except Exception as e:
        logger.error(f"Error Router: {e}")
        clasificacion = "otros"

    clasificacion = clasificacion.strip().upper()

    if clasificacion == "SALUDO":
        state["respuesta_final"] = "¡Hola! ¿En qué puedo ayudarte?"
        state["debug_pipeline"].append("[ROUTER] Detectado SALUDO.")
        state["destino"] = "fin"
        return state

    cat_final = "otros"
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
    
    filtro = {"category": cat} if cat != "otros" else None
    
    llm_fast = ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=MODELO_FAST, temperature=0.7)
    doc_hyde = generar_hyde(pregunta, llm_fast)
    state["debug_pipeline"].append(f"[BUSCADOR] HyDE imaginó: '{doc_hyde[:50]}...'")

    model_emb = SentenceTransformer(MODELO_EMBEDDINGS)
    q_emb = utils.generar_embeddings(model_emb, [doc_hyde])
    
    client_db = chromadb.PersistentClient(path=DB_DIR)
    col = client_db.get_collection(COLLECTION_NAME)
    
    res = col.query(
        query_embeddings=q_emb,
        n_results=5,     
        where=filtro 
    )
    
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

def nodo_generador(state: GraphState):
    """
    Nodo generador que redacta la respuesta final.
    """
    state["debug_pipeline"].append("[GENERADOR] Redactando respuesta...")
    contexto = "\n\n".join(state["contexto_docs"])
    
    if not contexto:
        state["respuesta_final"] = "Lo siento, no tengo información suficiente en mis guías para responder a tu pregunta."
        return state

    system_prompt = """
                    Eres un Asistente Virtual Especializado en Normativa para Autónomos en Bizkaia. 
                    Tu objetivo es resolver dudas sobre trámites, impuestos y ayudas basándote EXCLUSIVAMENTE en el contexto proporcionado.

                    DIRECTRICES:
                    1. EXCLUSIVIDAD DE DATOS: Responde únicamente utilizando la información del contexto. Si la respuesta no figura en los documentos, di exactamente: "Lo siento, no cuento con información específica en la documentación técnica para responder a esa duda".
                    2. RIGOR LOCAL: Prioriza siempre términos específicos de la Hacienda Foral de Bizkaia (ej. Batuz, TicketBai, Modelo 140, IAE).
                    3. ESTRUCTURA Y CLARIDAD:
                    - Usa un tono profesional, directo y alentador para el trabajador autónomo.
                    - Si el contexto incluye pasos de un trámite, preséntalos en una lista numerada.
                    4. CITACIÓN: Menciona el nombre del documento o guía de donde extraes la información (ej. "Según la Guía de Batuz 2024...") si está disponible en el contexto.
                    5. ADVERTENCIA LEGAL: Al final de respuestas sobre impuestos o trámites legales, añade una breve nota indicando que esta información es orientativa y recomienda consultar con la Hacienda Foral o una asesoría colegiada.
                    6. NO INVENTAR: No menciones ayudas estatales o de otras provincias si no aparecen en los fragmentos recuperados.
                    7. FORMATO DE RESPUESTA: Usa Markdown para dar formato a la respuesta, pudiendo usar negritas, cursivas, listas, etc. No uses emojis.
                    """

    user_prompt =   f"""
                    CONTEXTO RECUPERADO ({state.get('categoria_detectada', 'otros')}):
                    {contexto}
                    
                    PREGUNTA DEL USUARIO: "{state["pregunta"]}"
                    """

    llm = ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=MODELO_LLM, temperature=0.2)
    
    respuesta = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    state["respuesta_final"] = respuesta.content
    return state

def construir_grafo():
    workflow = StateGraph(GraphState)
    workflow.add_node("router", nodo_router)
    workflow.add_node("buscador", nodo_buscador)
    workflow.add_node("generador", nodo_generador)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["destino"],
        {
            "fin": END,
            "buscador": "buscador"
        }
    )
    workflow.add_edge("buscador", "generador")
    workflow.add_edge("generador", END)
    return workflow.compile()

app_graph = construir_grafo()
app = FastAPI(title="API RAG Autonomos Bizkaia", version="1.0")

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
def health(): return {"status": "ok", "version": "v1.0"}

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