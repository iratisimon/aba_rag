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
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer, util
from utilidades import utils
# import utils
import os
from loguru import logger
import json
from utilidades.funciones_preprocesado import leer_pdf
# from funciones_preprocesado import leer_pdf
import traceback
import uuid
import sys

load_dotenv()


DB_PATH = os.getenv("DB_PATH")
COLLECTION_NAME_PDFS = os.getenv("COLLECTION_NAME_PDFS")
COLLECTION_NAME_IMAGENES = os.getenv("COLLECTION_NAME_IMAGENES")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS")
MODELO_CLIP = os.getenv("MODELO_CLIP")
PDFS_DIR = utils.project_root() /"data"/"documentos"/"pdfs"

def obtener_coleccion(tipo="pdfs"):
    """
    Obtiene una colección de la base de datos (sin resetear).
    
    Args:
        tipo (str): Tipo de colección. Puede ser "pdfs" o "imagenes".
    
    Returns:
        chromadb.Collection: La colección solicitada.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    
    if tipo.lower() == "pdfs":
        return client.get_collection(COLLECTION_NAME_PDFS)
    elif tipo.lower() == "imagenes":
        return client.get_collection(COLLECTION_NAME_IMAGENES)
    else:
        raise ValueError(f"Tipo de colección no válido: {tipo}. Use 'pdfs' o 'imagenes'")

def cargar_modelos():
    """Carga todos los modelos de IA necesarios."""
    logger.info("Cargando modelos de IA...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Dispositivo detectado: {device.upper()}")

    # 1. Modelo de Embeddings (Texto)
    # 1. Modelo de Embeddings (Texto)
    logger.info("Cargando Embeddings de Texto...")
    model_emb = SentenceTransformer(MODELO_EMBEDDINGS, device=device)

    # 2. Modelo de Visión (CLIP)
    logger.info("Cargando CLIP (Visión)...")
    # Usamos transformers nativo para evitar error 'Image object not subscriptable' en SentenceTransformer
    model_clip = CLIPModel.from_pretrained(MODELO_CLIP).to(device)
    processor_clip = CLIPProcessor.from_pretrained(MODELO_CLIP)
    
    return model_emb, model_clip, processor_clip

def crear_db(reset=False):
    """
    Crea la base de datos de Chroma con dos colecciones (PDFs e imágenes).
    
    Args:
        reset (bool): Si es True, borra las colecciones existentes.
    
    Returns:
        dict: Diccionario con ambas colecciones {'pdfs': collection, 'imagenes': collection}
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME_PDFS)
            logger.info(f"Reset activado. Borrando colección '{COLLECTION_NAME_PDFS}'.")
        except:
            logger.warning(f"No se encontraba la colección '{COLLECTION_NAME_PDFS}'.")
        
        try:
            client.delete_collection(COLLECTION_NAME_IMAGENES)
            logger.info(f"Reset activado. Borrando colección '{COLLECTION_NAME_IMAGENES}'.")
        except:
            logger.warning(f"No se encontraba la colección '{COLLECTION_NAME_IMAGENES}'.")
        
        collection_pdfs = client.create_collection(COLLECTION_NAME_PDFS)
        collection_imagenes = client.create_collection(COLLECTION_NAME_IMAGENES)
    else:
        collection_pdfs = client.get_or_create_collection(COLLECTION_NAME_PDFS)
        collection_imagenes = client.get_or_create_collection(COLLECTION_NAME_IMAGENES)
    
    logger.info(f"Base de datos lista con colecciones: '{COLLECTION_NAME_PDFS}' e '{COLLECTION_NAME_IMAGENES}'")
    
    return {"pdfs": collection_pdfs, "imagenes": collection_imagenes}

def insertar_texto(texto, nombre_pdf, modelo_embeddings, collection, metadatos_json=None):
    """
    Inserta un texto en la base de datos con metadatos del PDF.
    
    Args:
        texto (str): El texto a insertar.
        nombre_pdf (str): El nombre del PDF.
        modelo_embeddings (SentenceTransformer): El modelo de embeddings.
        collection (chromadb.Collection): La colección de la base de datos.
        metadatos_json (dict): Metadatos del JSON de metadata_pdf.json
    """
    if not texto:
        return
    
    texto = utils.limpiar_texto(texto)
    chunks = utils.chunk_padre_hijo(texto)
    
    if not chunks:
        logger.warning(f"No se generaron chunks para {nombre_pdf}")
        return
    
    categoria = "sin_categoria"  # valor por defecto
    if metadatos_json:
        try:
            for doc in metadatos_json:
                if doc.get("archivo") == nombre_pdf:
                    categoria = doc.get("categoria", "sin_categoria")
                    break
        except Exception:
            logger.warning("Error leyendo metadatos de PDF proporcionados; usando categoria por defecto")

    textos_hijos = [item["texto_vectorizable"] for item in chunks]
    metadatas = []
    ids = []

    # Preparar metadatos para cada chunk (combinando JSON + chunk info)
    for idx, item in enumerate(chunks):
        metadatas.append({
            "source": nombre_pdf,
            "categoria": categoria,
            "type": "child",
            "parent_id": item["padre_id"],
            "contexto_expandido": item["texto_completo_padre"]
        })
        ids.append(f"{nombre_pdf}_child_{idx}")
            
    # Generar Embeddings (Solo de los HIJOS)
    embeddings = utils.generar_embeddings(modelo_embeddings, textos_hijos, batch_size=64)
        
    # Guardar en DB
    try:
        logger.info(f"  Insertando {len(ids)} chunks en ChromaDB...")
        collection.add(
            documents=textos_hijos,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Insertados {len(ids)} chunks de {nombre_pdf}")
    except Exception as e:
        logger.error(f"Error al insertar en ChromaDB: {str(e)}")
        logger.error(traceback.format_exc())

def insertar_imagen(model_clip, processor_clip, collection, metadata_imagenes=None):
    if not metadata_imagenes:
        logger.warning("No se proporcionaron metadatos de imágenes; nada que procesar")
        return

    logger.info(f"Procesando {len(metadata_imagenes)} imágenes desde metadatos")

    for meta in metadata_imagenes:
        try:
            ruta = Path(meta["ruta_imagen"])

            if not ruta.exists():
                logger.warning(f"Imagen no encontrada: {ruta}")
                continue

            image = Image.open(ruta).convert("RGB")

            # Procesar con CLIPProcessor
            inputs = processor_clip(images=image, return_tensors="pt").to(model_clip.device)
            
            with torch.no_grad():
                # get_image_features ya devuelve el tensor proyectado
                features = model_clip.get_image_features(**inputs)
                
                # Nos aseguramos de tener un vector (tensor) de PyTorch
                # get_image_features suele devolver el tensor proyectado, 
                # pero en algunas versiones devuelve un objeto BaseModelOutputWithPooling
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                elif hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif isinstance(features, (list, tuple)):
                    features = features[0]
                
                # Si a pesar de todo no es un tensor (caso raro), forzamos conversión si es posible
                if not isinstance(features, torch.Tensor) and hasattr(features, "__getitem__"):
                    try:
                        features = features[0]
                    except:
                        pass

                # Normalizar embeddings (crucial para CLIP)
                # Si llega aquí y no es tensor, fallará con un error descriptivo
                if not isinstance(features, torch.Tensor):
                    raise ValueError(f"No se pudo extraer el tensor de características. Tipo recibido: {type(features)}")

                features = features / features.norm(p=2, dim=-1, keepdim=True)
                embedding = features.cpu().numpy().tolist()[0] 

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding], # Embeddings debe ser una lista de listas [[...]]
                documents=[meta["nombre_archivo"]],
                metadatas=[{
                    "pdf_origen": meta["pdf_origen"],
                    "categoria": meta.get("categoria", "sin_categoria"),
                    "pagina": meta.get("pagina"),
                    "nombre_archivo": meta["nombre_archivo"],
                    "ruta_imagen": meta["ruta_imagen"],
                    "tipo": "imagen"
                }]
            )

        except Exception as e:
            logger.error(
                f"Error procesando imagen {meta.get('nombre_archivo')}: {e}"
            )

def main():
    """
    logger.info("\n RAG MULTIMODAL - CREANDO LA BASE DE DATOS \n")

    #Preguntar si borramos BD --> para ir haciendo pruebas
    resp = input("¿Borrar base de datos y empezar de cero? (s/n): ").lower()
    reset_db = (resp == 's')
    
    # Cargar modelos y crear db
    model_emb, model_clip, processor_clip = cargar_modelos()
    collections = crear_db(reset_db)
    
    # Cargar metadatos de PDFs una sola vez
    metadatos_pdf = []
    try:
        with open("data/metadata_pdf.json", "r", encoding="utf-8") as f:
            metadatos_pdf = json.load(f)
    except FileNotFoundError:
        logger.warning("No se encontró data/metadata_pdf.json; se usará categoria por defecto para todos los PDFs")
    except json.JSONDecodeError:
        logger.error("Error al parsear data/metadata_pdf.json; se usará categoria por defecto para todos los PDFs")

    # Procesar pdfs
    pdfs = list(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error(f"Error: No hay PDFs en {PDFS_DIR}")
        return
    
    for pdf in pdfs:
        nombrePDF = pdf.name
        logger.info(f"Procesando: {nombrePDF}...")
        try:
            # Usar función de preprocesado
            texto_completo = leer_pdf(str(pdf))
            # Insertar texto (usar metadatos cargados en main)
            insertar_texto(texto_completo, nombrePDF, model_emb, collections["pdfs"], metadatos_pdf)
        except Exception as e:
            logger.error(f"Error procesando {nombrePDF}: {e}")
            continue

    
    # Cargar metadatos de imágenes una sola vez
    metadata_imagenes = []
    try:
        with open("data/metadata_imagenes.json", "r", encoding="utf-8") as f:
            metadata_imagenes = json.load(f)
    except FileNotFoundError:
        logger.warning("No se encontró data/metadata_imagenes.json; no se procesarán imágenes")
    except json.JSONDecodeError:
        logger.error("Error al parsear data/metadata_imagenes.json; no se procesarán imágenes")
    
    # Procesar imágenes
    logger.info("\nProcesando imágenes...")
    insertar_imagen(
        model_clip=model_clip,
        processor_clip=processor_clip,
        collection=collections["imagenes"],
        metadata_imagenes=metadata_imagenes
    )
    logger.info("\n PROCESAMIENTO TERMINADO")
    logger.info(f"Base de datos guardada en: {DB_PATH}")"""
    
    logger.info("\n RAG MULTIMODAL - ACTUALIZANDO COLECCIÓN DE IMÁGENES \n")

    # Cargar modelos
    # Cargar modelos
    _, model_clip, processor_clip = cargar_modelos()  # Ignoramos embeddings de texto porque no tocamos PDFs

    # Conectar con la base de datos
    client = chromadb.PersistentClient(path=DB_PATH)

    # Borrar colección de imágenes si existe
    try:
        client.delete_collection(COLLECTION_NAME_IMAGENES)
        logger.info(f"Colección '{COLLECTION_NAME_IMAGENES}' eliminada.")
    except Exception as e:
        logger.warning(f"No se pudo eliminar '{COLLECTION_NAME_IMAGENES}': {e}")

    # Crear nueva colección de imágenes
    collection_imagenes = client.create_collection(COLLECTION_NAME_IMAGENES)
    logger.info(f"Colección '{COLLECTION_NAME_IMAGENES}' creada de nuevo.")

    # Cargar metadatos de imágenes
    metadata_imagenes = []
    try:
        with open("data/metadata_imagenes.json", "r", encoding="utf-8") as f:
            metadata_imagenes = json.load(f)
    except FileNotFoundError:
        logger.warning("No se encontró data/metadata_imagenes.json; no se procesarán imágenes")
        return
    except json.JSONDecodeError:
        logger.error("Error al parsear data/metadata_imagenes.json; no se procesarán imágenes")
        return

    # Procesar e insertar imágenes usando CLIP
    logger.info("\nProcesando imágenes...")
    insertar_imagen(
        model_clip=model_clip,
        processor_clip=processor_clip,
        collection=collection_imagenes,
        metadata_imagenes=metadata_imagenes
    )

    logger.info("\nPROCESAMIENTO TERMINADO")
    logger.info(f"Base de datos guardada en: {DB_PATH}")

if __name__ == "__main__":
    main()