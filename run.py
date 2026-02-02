import subprocess
import time
import sys
import os
from loguru import logger

def start():
    # 1. Arrancar FastAPI en segundo plano
    logger.info("Iniciando API de Bizkaia (FastAPI)...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "src.api.api:app", 
        "--host", "127.0.0.1", "--port", "8000"
    ])

    # 2. Esperar a que los modelos se carguen
    logger.info("Esperando a que los modelos carguen en memoria...")
    time.sleep(20)

    # 3. Arrancar Streamlit
    logger.info("Iniciando Interfaz (Streamlit)...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/interfaz.py"])
    except KeyboardInterrupt:
        logger.info("Cerrando servicios...")
    finally:
        api_process.terminate()
        logger.info("Procesos finalizados.")

if __name__ == "__main__":
    start()