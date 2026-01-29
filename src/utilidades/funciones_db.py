"""
Este script contiene las funciones relacionadas con la base de datos.
Utiliza ChromaDB como base de datos.
"""

import chromadb
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
import torch
import fitz
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import utils
import os
from loguru import logger

load_dotenv()

DB_PATH = os.getenv("DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS")
MODELO_BLIP = os.getenv("MODELO_BLIP")

def crear_db(reset=False):
    """
    Crea la base de datos de Chroma.
    
    Args:
        reset (bool): Si es True, borra la base de datos existente.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Reset activado. Borrando la base de datos existente.")
        except:
            logger.warning("No se encontraba la base de datos existente.")
        collection = client.create_collection(COLLECTION_NAME)
    else:
        collection = client.get_or_create_collection(COLLECTION_NAME)
    
    return collection

def insertar_texto(texto, nombre_pdf, modelo_embeddings, collection):
    """
    Inserta un texto en la base de datos.
    
    Args:
        texto (str): El texto a insertar.
        nombre_pdf (str): El nombre del PDF.
        modelo_embeddings (SentenceTransformer): El modelo de embeddings.
        collection (chromadb.Collection): La colección de la base de datos.
    """

    if not texto:
        return
    
    texto = utils.limpiar_texto(texto)
    chunks = utils.chunk(texto, 500, 10)

    ids = []
    metadatas = []
    embeddings = []
    documentos = []

    for i, chunk in enumerate(chunks):
        emb = modelo_embeddings.encode(chunk, normalize_embeddings=True).tolist()
        ids.append(f"{nombre_pdf}_chnk_{i}")
        metadatas.append({"pdf": nombre_pdf})
        embeddings.append(emb)
        documentos.append(chunk)

    collection.add(
        ids=ids,
        metadatas=metadatas,
        embeddings=embeddings,
        documents=documentos
    )
