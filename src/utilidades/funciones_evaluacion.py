import re
import logging
from typing import Tuple
from utilidades import prompts

logger = logging.getLogger("evaluacion_rag")

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
