"""En este script aparecen la sfunciones relacionadas con el preprocesado de los datos 
que se van a recoger en la base de datos"""
import json
import os
from loguru import logger
from utilidades import utils
import glob
import fitz
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Lista de categorías
CATEGORIAS_VALIDAS = [
    "Laboral",      
    "Fiscal", 
    "Ayudas_y_Subvenciones"                
]
# Modelo
MODELO_CLASIFICADOR = os.getenv("MODELO_FAST", "llama-3.1-8b-instant")

# Configuración de Rutas usando utils
DATA_DIR = utils.project_root() /"data"/"documentos"
PDF_DIR = utils.project_root() /"data"/"documentos"/"pdfs"
IMAGENES_DIR = utils.project_root() /"data"/"documentos"/"imagenes"
logger.info(f"Ruta de datos: {DATA_DIR}")

# Configuración LLM para clasificación automática
LLM_API_KEY = os.getenv("LLM_API_KEY", "groq")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")

def  leer_pdf(ruta_pdf: str) -> str:
    """
    Lee todo el texto de un archivo PDF.
    
    Args:
        ruta_pdf: Ruta absoluta o relativa al archivo PDF.
        
    Returns:
        String con todo el texto del PDF concatenado.
    """
    if not os.path.exists(ruta_pdf):
        logger.error(f"No encuentro el archivo: {ruta_pdf}")
        return ""

    logger.info(f"  Leyendo texto del PDF...")
    doc = fitz.open(ruta_pdf)
    texto_completo = ""
    
    for pagina in doc:
        texto_completo += pagina.get_text() + "\n"
        
    logger.info(f"    ✓ Leídas {len(doc)} páginas ({len(texto_completo)} caracteres)")
    return texto_completo

def extraer_imagen(ruta_pdf):
    """
    Extrae todas las imágenes de un PDF y las guarda en disco.
    Las imágenes heredan la categoría del PDF padre.
    """
    doc = fitz.open(ruta_pdf)
    nombre_base = os.path.basename(ruta_pdf).replace('.pdf', '')

    # Recorrer páginas
    for num_pag, pagina in enumerate(doc, 1):
        for i, img in enumerate(pagina.get_images(full=True)):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                bytes_img = base["image"]
                ext = base["ext"]
                
                nombre_archivo = f"{nombre_base}-{num_pag}-{i}.{ext}"
                ruta_completa = IMAGENES_DIR / nombre_archivo
                
                # Guardar en disco
                with open(ruta_completa, "wb") as f:
                    f.write(bytes_img)
            except Exception as e:
                pass # Ignorar errores puntuales de extracción

def clasificar_documento(nombre_archivo: str, texto_inicio: str, client_llm) -> str:
    """
    Clasifica un documento usando LLM.
    """
    system_prompt = f"""Eres un sistema de clasificación documental para una aplicación RAG legal-administrativa
dirigida a personas autónomas en Bizkaia.
Categorías válidas: {', '.join(CATEGORIAS_VALIDAS)}

REGLAS:
- Responde SOLO con el nombre exacto de la categoría.
- Usa EXCLUSIVAMENTE una de las categorías permitidas.
- NO inventes nuevas categorías.
- NO devuelvas explicaciones ni texto adicional.
- Si el documento contiene varios temas, elige el tema principal.
- Si ninguna categoría encaja claramente, usa "otros".
- NO expliques tu razonamiento."""

    user_prompt = f"""Clasifica este documento:

ARCHIVO: {nombre_archivo}
TEXTO:
"{texto_inicio[:1000]}"

CATEGORIA:"""

    try:
        resp = client_llm.chat.completions.create(
            model=MODELO_CLASIFICADOR,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        categoria_raw = resp.choices[0].message.content.strip()
        
        for cat in CATEGORIAS_VALIDAS:
            if cat.lower() in categoria_raw.lower():
                return cat
        
        logger.warning(f"  LLM respondió '{categoria_raw}' (no válida) → Usando 'otros'")
        return "General"
        
    except Exception as e:
        logger.warning(f"Fallo al clasificar {nombre_archivo}: {e}")
        return "General"
    
def cargar_metadata_pdfs(
    nombre_archivo: str,
    categoria: str,
) -> None:
    metadata_file = utils.project_root() / "data" / "metadata_pdf.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # Si existe, lo cargamos; si no, empezamos lista vacía
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Evitar duplicados (por nombre de archivo)
    data = [item for item in data if item.get("archivo") != nombre_archivo]

    # Añadir / actualizar
    data.append({
        "archivo": nombre_archivo,
        "categoria": categoria
    })

    # Guardar
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generar_metadatos_imagenes_existentes() -> list:
    """
    Genera metadatos SOLO para las imágenes que realmente existen en disco.
    Lee las categorías directamente del archivo metadata_pdfs.json.
    Esto permite borrar manualmente imágenes no deseadas antes de generar metadatos.
    
    Returns:
        Lista de metadatos solo de imágenes existentes
    """
    # 1. Cargar categorías desde el JSON de PDFs
    metadata_pdfs_file = utils.project_root() / "data" / "metadata_pdf.json"
    
    if not metadata_pdfs_file.exists():
        logger.error("No se encontró metadata_pdfs.json. Ejecuta primero el preprocesado de PDFs.")
        return []
    
    with open(metadata_pdfs_file, "r", encoding="utf-8") as f:
        pdfs_data = json.load(f)
    
    # Crear diccionario {nombre_pdf: categoria}
    categoria_por_pdf = {item["archivo"]: item["categoria"] for item in pdfs_data}
    
    # 2. Buscar imágenes en disco
    if not IMAGENES_DIR.exists():
        logger.warning(f"No existe la carpeta de imágenes: {IMAGENES_DIR}")
        return []
    
    metadatos_imagenes = []
    imagenes_encontradas = list(IMAGENES_DIR.glob("*.*"))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generando metadatos de imágenes existentes...")
    logger.info(f"{'='*60}")
    logger.info(f"Imágenes encontradas en disco: {len(imagenes_encontradas)}\n")
    
    for ruta_img in imagenes_encontradas:
        nombre_archivo = ruta_img.name
        
        try:
            partes = nombre_archivo.split('-')
            if len(partes) < 2:
                logger.warning(f"Formato inesperado: {nombre_archivo}")
                continue
            
            nombre_base = partes[0]  # nombre del PDF sin extensión
            num_pag = partes[1]  # "NUM-contexto.ext"
            
            # Buscar el PDF origen y su categoría desde los metadatos
            pdf_origen = f"{nombre_base}.pdf"
            categoria = categoria_por_pdf.get(pdf_origen)
            
            if not categoria:
                logger.warning(f"No se encontró categoría para {pdf_origen}, usando 'Ayudas_y_Subvenciones'")
                categoria = "Ayudas_y_Subvenciones"
            
            metadatos_imagenes.append({
                "ruta_imagen": str(ruta_img),
                "nombre_archivo": nombre_archivo,
                "pdf_origen": pdf_origen,
                "categoria": categoria,
                "pagina": int(num_pag)
            })
            
            logger.info(f"  ✓ [{categoria}] {nombre_archivo}")
            
        except Exception as e:
            logger.warning(f"  ✗ Error procesando {nombre_archivo}: {e}")
    
    logger.info(f"\n  Total metadatos generados: {len(metadatos_imagenes)}")
    return metadatos_imagenes

def cargar_metadata_imagenes(metadatos_imagenes: list) -> None:
    """Guarda metadatos de imágenes en JSON"""
    metadata_file = utils.project_root() / "data" / "metadata_imagenes.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Añadir nuevas imágenes
    for img_meta in metadatos_imagenes:
        # Evitar duplicados por nombre de archivo
        data = [item for item in data if item.get("nombre_archivo") != img_meta["nombre_archivo"]]
        data.append(img_meta)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    """
    logger.info("Iniciando pre-procesado")
    
    patron_ruta = os.path.join(PDF_DIR, "*.pdf")
    archivos = glob.glob(patron_ruta)
    
    # Setup Cliente LLM
    client_llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    if not archivos:
        logger.error("No se encontraron archivos PDF en la carpeta.")
        return
    
    logger.info(f"Encontrados {len(archivos)} PDFs para procesar\n")
    
    for archivo in archivos:
        nombre_archivo = os.path.basename(archivo)
        logger.info(f"Procesando: {nombre_archivo}")
        
        # 1. Leer texto del PDF
        texto = leer_pdf(archivo)
        
        # 2. Clasificar documento
        categoria = clasificar_documento(nombre_archivo, texto, client_llm)
        logger.info(f"  [Clasificación] → {categoria}")
        
        # 3. Guardar metadatos del PDF
        cargar_metadata_pdfs(nombre_archivo,categoria)
        
        # 4. Extraer TODAS las imágenes (sin metadatos aún)
        extraer_imagen(archivo)"""
    
    # Generar metadatos solo de imágenes existentes
    metadatos_imgs = generar_metadatos_imagenes_existentes()
        
    # Guardar metadatos
    if metadatos_imgs:
        cargar_metadata_imagenes(metadatos_imgs)
        logger.info(f"\n✓ Metadatos guardados: {len(metadatos_imgs)} imágenes")

if __name__ == "__main__":
    main()
