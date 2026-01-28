"""
Este script contiene las funciones relacionadas con la base de datos.
Utiliza ChromaDB como base de datos.
"""

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image
import utils
import load_dotenv
import os
from loguru import logger

load_dotenv()

def crear_db():
    """
    Crea la base de datos de Chroma.
    """
    db_path = utils.project_root() / os.getenv("DB_PATH", "chroma_db")
    collection_texto = os.getenv("COLLECTION_NAME", "autonomos_bizkaia_texto")
    collection_imagenes = os.getenv("COLLECTION_IMAGENES", "autonomos_bizkaia_imagenes")
    
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    client = chromadb.PersistentClient(path=db_path)

    client.get_or_create_collection(name=collection_texto)
    
    logger.info("Base de datos creada con éxito")