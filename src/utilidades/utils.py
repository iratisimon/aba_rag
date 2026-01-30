from pathlib import Path 
import re
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def generar_embeddings(model, textos: List[str]) -> List[List[float]]:
    """Genera embeddings usando el modelo SentenceTransformer dado."""
    return model.encode(textos, normalize_embeddings=True).tolist()

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def limpiar_texto(texto: str) -> str:
    """
    Limpieza SUAVE para texto extraído de PDF, pensada para embeddings.
    Objetivo: quitar ruido de extracción SIN perder información técnica.

    Limpieza suave para PDFs (quitar ruido sin destruir semántica, los embeddings (Qwen) entienden tildes, mayúsculas y contexto)
    No queremos destruir esa información. Solo queremos quitar el "ruido de PDF" (guiones partidos, espacios raros).
    """
    if not texto:
        return ""

    #Convertir todos los retornos de carro (\r) en saltos de línea (\n) y los tabs (\t) en espacios.
    t = texto.replace("\r", "\n").replace("\t", " ")

    # Unir palabras cortadas por guion al final de línea:
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Convertir saltos de línea “sueltos” en espacios (sin cargarse párrafos)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)

    # Sustituir caracteres como el espacio no separable (\u00A0) por espacios normales.
    t = t.replace("\u00A0", " ")

    #Compactar espacios
    t = re.sub(r"[ ]{2,}", " ", t).strip()

    return t

def chunk(texto: str, chunk_size=200, overlap=50) -> list:
    """
    Usa LangChain para dividir el texto en chunks.

    args:
        texto: str
        chunk_size: int
        overlap: int
    returns:
        list de chunks
    """

    logger.info(f" Configurando Splitter: Size={chunk_size}, Overlap={overlap}")
    
    # Crear el objeto splitter 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Tamaño objetivo
        chunk_overlap=overlap,      # Solapamiento para contexto
        length_function=len,        # Cómo medimos (por caracteres)
        separators=["\n\n", "\n", " ", ""] # Prioridad de corte : Párrafos > Líneas > Frases 
    )

    # Ejecutar el corte
    chunks = text_splitter.split_text(texto)
    
    return chunks