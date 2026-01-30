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
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from utilidades import utils
import os
from loguru import logger
import json
from utilidades.funciones_preprocesado import leer_pdf

load_dotenv()

DB_PATH = os.getenv("DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS")
MODELO_BLIP = os.getenv("MODELO_BLIP")
JSON_DIR = utils.project_root() /"data"
PDFS_DIR = utils.project_root() /"data"/"documentos"/"pdfs"
IMAGENES_DIR = utils.project_root() /"data"/"documentos"/"imagenes"

def obtener_coleccion():
    """
    Obtiene la colección de la base de datos (sin resetear).
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)

def cargar_modelos():
    """Carga todos los modelos de IA necesarios."""
    logger.info("Cargando modelos de IA... (Esto puede tardar un poco)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Dispositivo detectado: {device.upper()}")

    # 1. Modelo de Embeddings (Texto)
    logger.info("   Cargando Embeddings...")
    model_emb = SentenceTransformer(MODELO_EMBEDDINGS)

    # 2. Modelo de Visión (BLIP)
    logger.info("   Cargando BLIP (Visión)...")
    processor = BlipProcessor.from_pretrained(MODELO_BLIP)
    model_blip = BlipForConditionalGeneration.from_pretrained(MODELO_BLIP)
    model_blip.to(device)
    
    
    return model_emb, processor, model_blip, device

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

def insertar_texto(texto, nombre_pdf, modelo_embeddings, collection, metadatos_json=None):
    """
    Inserta un texto en la base de datos.
    
    Args:
        texto (str): El texto a insertar.
        nombre_pdf (str): El nombre del PDF.
        modelo_embeddings (SentenceTransformer): El modelo de embeddings.
        collection (chromadb.Collection): La colección de la base de datos.
        metadatos_json (dict): Metadatos del JSON de preprocesado.
    """
    if not texto:
        return
    
    texto = utils.limpiar_texto(texto)
    chunks = utils.chunk(texto, 500, 10) #cambiar esto

    ids = []
    metadatas = []
    embeddings = []
    documentos = []

    for i, chunk in enumerate(chunks):
        emb = modelo_embeddings.encode(chunk, normalize_embeddings=True).tolist()
        ids.append(f"{nombre_pdf}_chnk_{i}")
        
        # Usar metadatos del JSON + info del chunk
        meta = metadatos_json.copy() if metadatos_json else {}
        meta.update({"pdf": nombre_pdf, "chunk": i, "tipo": "texto"})
        metadatas.append(meta)
        
        embeddings.append(emb)
        documentos.append(chunk)

    collection.add(
        ids=ids,
        metadatas=metadatas,
        embeddings=embeddings,
        documents=documentos
    )

def insertar_imagen(lista_imagenes, nombre_pdf, processor, model_blip, model_emb, device, collection, metadatos_json=None):
    """
    Procesa imágenes y las inserta con metadatos enriquecidos del JSON.
    """
    if not lista_imagenes:
        return 0
    
    count = 0
    ids = []
    embs = []
    docs = []
    metas = []
    
    logger.info(f"Procesando {len(lista_imagenes)} imágenes con I.A...")
    
    for img_data in tqdm(lista_imagenes):
        try:
            imagen_pil = Image.open(img_data["ruta"]).convert('RGB')
            inputs = processor(imagen_pil, return_tensors="pt").to(device)
            out = model_blip.generate(**inputs, max_length=100)
            descripcion_ingles = processor.decode(out[0], skip_special_tokens=True)
            vector = model_emb.encode(descripcion_ingles, normalize_embeddings=True).tolist()
            
            ids.append(f"{img_data['nombre_archivo'].split('.')[0]}")
            embs.append(vector)
            docs.append(descripcion_ingles)
            
            # Usar metadatos del JSON directamente
            meta = {
                "pdf_origen": img_data.get("pdf_origen", nombre_pdf),
                "categoria": img_data.get("categoria", ""),
                "pagina": img_data.get("pagina", 0),
                "indice": img_data.get("indice", 0),
                "nombre_archivo": img_data.get("nombre_archivo", ""),
                "tipo": "imagen"
            }
            metas.append(meta)
            count += 1
            
        except Exception as e:
            logger.error(f"Error procesando imagen {img_data['nombre_archivo']}: {e}")
            
    if count > 0:
        collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        logger.info(f"   ✓ {count} imágenes insertadas")
        
    return count

def main():
    print("\n RAG MULTIMODAL - PROCESAMIENTO SIN CLASES \n")
    """
    # 1. Buscar PDFs
    pdfs = list(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Error: No hay PDFs en {PDFS_DIR}")
        return
    
    # 2. Preguntar si borramos BD
    resp = input("¿Borrar base de datos y empezar de cero? (s/n): ").lower()
    reset_db = (resp == 's')"""
    
    # 3. Cargar todo
    model_emb, processor, model_blip, device = cargar_modelos()
    #collection = crear_db(reset_db)
    collection = crear_db(reset=False)
    
    # 4. Cargar metadatos del preprocesado
    #metadata_pdf = JSON_DIR / "metadata_pdf.json"  
    metadata_imagenes = JSON_DIR / "metadata_imagenes.json" 
    """
    if not metadata_pdf.exists() or not metadata_imagenes.exists():
        logger.error("Faltan archivos de metadatos")
        return
    
    with open(metadata_pdf, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    # Convertir lista a dict usando "archivo" como clave
    metadatos_pdfs = {item['archivo']: item for item in metadata_list if 'archivo' in item}"""

    if not metadata_imagenes.exists():
        logger.error(f"No se encontró {metadata_imagenes}")
        return
    
    with open(metadata_imagenes, 'r', encoding='utf-8') as f:
        imagenes_list = json.load(f)
    
    # Para imágenes usa "nombre_archivo"
    metadatos_imagenes = {item['nombre_archivo']: item for item in imagenes_list if 'nombre_archivo' in item}
    """
    # 5. Procesar uno a uno
    for pdf in pdfs:
        nombre = pdf.name
        logger.info(f"Procesando: {nombre}...")
        
        try:
            # Usar función de preprocesado
            texto_completo = leer_pdf(str(pdf))
            
            # Obtener metadatos del JSON para este PDF
            metadatos_pdf = metadatos_pdfs.get(nombre, {})
            
            # Insertar texto
            insertar_texto(texto_completo, nombre, model_emb, collection, metadatos_pdf)
        except Exception as e:
            logger.error(f"Error procesando {nombre}: {e}")
            continue"""

    # 6. Procesar todas las imágenes de la carpeta
    logger.info("\nProcesando imágenes...")
    imagenes = list(IMAGENES_DIR.glob("*"))
    
    if imagenes:
        lista_imagenes = []
        
        for img in imagenes:
            # Obtener metadatos de esta imagen del JSON
            metadatos_img = metadatos_imagenes.get(img.name, {})
            
            lista_imagenes.append({
                "ruta": str(img),
                "nombre_archivo": img.name,
                "pdf_origen": metadatos_img.get("pdf_origen", ""),
                "categoria": metadatos_img.get("categoria", ""),
                "pagina": metadatos_img.get("pagina", 0),
                "indice": metadatos_img.get("indice", 0)
            })
        
        # Procesar todas las imágenes juntas
        insertar_imagen(lista_imagenes, "todas_las_imagenes", processor, model_blip, model_emb, device, collection, {})
    
    logger.info("\n PROCESAMIENTO TERMINADO")
    logger.info(f"Base de datos guardada en: {DB_PATH}")


if __name__ == "__main__":
    main()