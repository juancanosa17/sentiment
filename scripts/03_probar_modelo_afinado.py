# 📁 scripts/03_probar_modelo_afinado.py (Versión Actualizada)

from transformers import pipeline
import pandas as pd
import os

# Directorio del modelo (espera que esté en la raíz del proyecto)
model_path = "./modelo_afinad"

# Nuevos comentarios sarcásticos para probar la generalización del modelo
nuevos_sarcasmos = [
    # Ejemplo 1: Batería que dura poco
    "Me encanta cómo la batería de este teléfono dura unos impresionantes treinta minutos. Pura innovación.",
    
    # Ejemplo 2: Instrucciones inútiles
    "El manual de instrucciones era tan claro que preferí armar el mueble adivinando. Una experiencia muy intuitiva.",

    # Ejemplo 3: Envío incorrecto
    "Recibir la talla incorrecta después de un mes de espera fue, sinceramente, el punto culminante de mi semana."
]

print(f"Probando {len(nuevos_sarcasmos)} nuevos comentarios sarcásticos...")

# Verificar si el modelo existe antes de cargarlo
if not os.path.exists(model_path):
    print(f"Error: No se encontró el directorio del modelo en '{model_path}'")
    print("Asegúrate de haber descargado y descomprimido el artefacto del modelo entrenado.")
else:
    # Cargar el pipeline usando TU modelo afinado
    sentiment_analyzer_afinad = pipeline(
        "text-classification",
        model=model_path 
    )

    # Analizar los nuevos comentarios
    resultados = sentiment_analyzer_afinad(nuevos_sarcasmos)
    
    # Mostrar los resultados en una tabla
    df = pd.DataFrame({
        'Comentario': nuevos_sarcasmos,
        'Sentimiento Predicho': [res['label'] for res in resultados],
        'Confianza': [round(res['score'], 4) for res in resultados]
    })

    print("\n--- Resultados de la Prueba ---")
    print(df.to_string(index=False))
