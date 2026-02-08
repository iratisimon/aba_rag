import re
import os
import logging
import random
import time
import json
from typing import List, Optional
from utilidades import prompts, utils

logger = logging.getLogger("evaluacion_rag")
# Definir la ruta del golden set usando la variable de entorno si está disponible
GOLDEN_SET_FILE = os.getenv(
    "GOLDEN_SET_FILE",
    str(utils.project_root() / "src" / "golden_set_automatico.jsonl")
)

def evaluar_fidelidad(pregunta: str, respuesta: str, contexto: str, client_llm, model_name: str) -> int:
    """
    JUEZ DE FIDELIDAD (Faithfulness) - Detector de Alucinaciones
    """
    
    prompt_juez = prompts.obtener_prompt_fidelidad(contexto, respuesta)

    try:
        if hasattr(client_llm, "invoke"):
            response = client_llm.invoke([{"role": "user", "content": prompt_juez}])
            texto = response.content.strip()
        else:
            veredicto = client_llm.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_juez}],
                temperature=0,
                max_tokens=5
            )
            texto = veredicto.choices[0].message.content.strip()

        match = re.search(r'\b([01])\b', texto)
        if match:
            return int(match.group(1))

        if "1" in texto: return 1
        if "0" in texto: return 0
        
        return 0
    except Exception as e:
        logger.error(f"Error en evaluación de fidelidad: {e}")
        return 0

def evaluar_relevancia(pregunta: str, respuesta: str, client_llm, model_name: str) -> int:
    """
    JUEZ DE RELEVANCIA (Answer Relevance)
    """
    
    prompt_juez = prompts.obtener_prompt_relevancia(pregunta, respuesta)

    try:
        if hasattr(client_llm, "invoke"):
            response = client_llm.invoke([{"role": "user", "content": prompt_juez}])
            texto = response.content.strip()
        else:
            veredicto = client_llm.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_juez}],
                temperature=0,
                max_tokens=5
            )
            texto = veredicto.choices[0].message.content.strip()

        # Buscar primer dígito VÁLIDO (1-5)
        for char in texto:
            if char.isdigit():
                num = int(char)
                if 1 <= num <= 5:
                    return num
        return 1
    except Exception as e:
        logger.error(f"Error en evaluacion de relevancia: {e}")
        return 1
    
def generar_pregunta_para_chunk(texto_chunk: str, metadata: dict, client_llm, model_name: str):
    """Usa el LLM para generar una pregunta basada en el texto y su contexto (metadata).
    """

    prompt_juez = prompts.obtener_prompt_eval_retrieval(texto_chunk)
    user_content = f"TEXTO:\n{texto_chunk[:1500]}\n\nPREGUNTA:"""

    logger.info(f"Generando pregunta para chunk (meta={metadata.get('source','-')}): {user_content[:200]}...")

    try:
        if hasattr(client_llm, "invoke"):
            response = client_llm.invoke([{"role": "system", "content": prompt_juez}, {"role": "user", "content": user_content}])
            texto = response.content.strip()
        else:
            veredicto = client_llm.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": prompt_juez}, {"role": "user", "content": user_content}],
                temperature=0.7
            )
            texto = veredicto.choices[0].message.content.strip()

        # Normalizar
        pregunta = texto.strip().strip('\"').strip()
        # Asegurarnos que termina en signo interrogación o en formato pregunta
        if len(pregunta) < 5:
            return None
        return pregunta
    except Exception as e:
        logger.error(f"Error generando pregunta: {e}")
        return None
    
def crear_golden_set_automatico(collection, client_llm, model_name: str, num_preguntas: int = 20):
    """Crea un dataset de evaluación automáticamente (Formato Estándar).
    """
    logger.info(f"Generando Golden Set Automático ({num_preguntas} preguntas)...")

    all_data = collection.get()
    all_ids = all_data.get('ids', [])
    all_docs = all_data.get('documents', [])
    all_metas = all_data.get('metadatas', [])

    if len(all_ids) == 0:
        logger.error("La colección está vacía.")
        return []

    indices = random.sample(range(len(all_ids)), min(num_preguntas, len(all_ids)))
    golden_set = []

    for i, idx in enumerate(indices):
        chunk_id = all_ids[idx]
        texto = all_docs[idx]
        meta = all_metas[idx] if idx < len(all_metas) else {}

        logger.info(f"Generando pregunta {i+1} de {num_preguntas} -> {pregunta}")
        pregunta = generar_pregunta_para_chunk(texto, meta, client_llm, model_name)

        if pregunta:
            logger.info(f"[{i+1}/{num_preguntas}] Generada: {pregunta[:60]}...")
            entry = {
                "id": f"q_{i}",
                "query": pregunta,
                "relevant_ids": [chunk_id],
                "texto_original": texto[:200],
                "metadata": {
                    "source": meta.get('source', 'unknown'),
                    "category": meta.get('category', 'General')
                }
            }
            golden_set.append(entry)
            time.sleep(2)

        else:
            logger.warning(f"No se pudo generar pregunta para chunk id={chunk_id}")

    # Guardar en archivo newline-delimited JSON
    try:
        with open(GOLDEN_SET_FILE, 'w', encoding='utf-8') as f:
            for entry in golden_set:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Golden Set guardado en: {GOLDEN_SET_FILE}")
    except Exception as e:
        logger.error(f"Error guardando golden set: {e}")

    return golden_set

def evaluar_retrieval(collection, model_emb, golden_set, top_k: int = 3):
    """
    Evalúa la calidad del módulo de recuperación.
    
    Args:
        collection: Colección de ChromaDB con los documentos indexados
        model_emb: Modelo de embeddings (SentenceTransformer)
        golden_set: Lista de casos de prueba {query, relevant_ids}
        top_k (int): Número de documentos a recuperar por consulta.
    
    Returns:
        tuple: (hit_rate, mrr) métricas de evaluación
    """

    logger.info(f"\n Iniciando Evaluación (Top-{top_k})...")
    
    aciertos = 0
    mrr_sum = 0.0
    
    # Recorrer cada pregunta del golden set
    for i, item in enumerate(golden_set):
        pregunta = item.get('query')
        target_ids = item.get('relevant_ids', [])

        # 1. Generar embedding de la pregunta
        query_emb = utils.generar_embeddings(model_emb, [pregunta])

        # 2. Consultar la colección (ChromaDB)
        results = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )

        recuperados_ids = results.get('ids', [[]])[0]
        
        # 2. Comprobar si acertamos (Si ALGUNO de los targets está en los recuperados)
        # Intersección de listas > 0

        # Comprueba si la respuesta correcta está dentro de los 3 que trajo el buscador.
        acierto = any(tid in recuperados_ids for tid in target_ids)
        
        if acierto:
            aciertos += 1
            # Para MRR, buscamos la posición del PRIMER acierto.  Si acertamos, miramos en qué puesto quedó.
            # Si quedó el 1º -> Sumamos 1/1 = 1.0 puntos.
            # Si quedó el 2º -> Sumamos 1/2 = 0.5 puntos.
            # Premia acertar Arriba del todo. Acertar el 3º es bueno, pero acertar el 1º es mejor.
            for rank, rid in enumerate(recuperados_ids):
                if rid in target_ids:
                    mrr_sum += 1.0 / (rank + 1)
                    logger.info(f"    Acierto (Pos {rank+1}): {pregunta[:50]}...")
                    break
        else:
            logger.info(f"    Fallo: {pregunta[:50]}...")
            
    # 3. Calcular Métricas Finales
    #	total: Número de preguntas
    #	aciertos: Cuántas veces encontró el chunk correcto
    #	mrr_sum:	Suma de 1/posición de cada acierto
    
    total = len(golden_set)
    hit_rate = aciertos / total if total > 0 else 0
    mrr = mrr_sum / total if total > 0 else 0
    
    return hit_rate, mrr