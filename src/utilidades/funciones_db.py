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

def crear_db(db_path: str = os.getenv("DB_PATH"), collection_name: str = os.getenv("COLLECTION_NAME")):
    """
    Crea la base de datos de Chroma.
    """

    logger.info(f"Creando base de datos en {db_path} con colección {collection_name}")
    # 1. Configurar el cliente de Chroma
    client = chromadb.PersistentClient(path=utils.project_root() / db_path)

    # 2. Configurar la función de embedding multimodal (CLIP)
    # Esto permite que texto e imágenes compartan el mismo "espacio"
    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()

    # 3. Crear la colección multimodal
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        data_loader=image_loader
    )

    logger.info(f"Base de datos creada con éxito en {db_path} con colección {collection_name}")

    # --- EJEMPLO DE CARGA ---

    # Añadir un texto (ej. normativa del IAE)
    collection.add(
        ids=["doc_1"],
        documents=["Guía sobre el TicketBai en Bizkaia para autónomos..."]
    )

    # Añadir una imagen (ej. captura de un formulario de la Diputación)
    # Nota: Debes pasar la ruta de la imagen
    collection.add(
        ids=["img_1"],
        uris=["./imagenes/formulario_modelo_036.png"]
    )