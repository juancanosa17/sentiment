# 📁 scripts/01_crear_dataset.py (Versión Corregida)

import pandas as pd
import csv # <--- AÑADE ESTA LÍNEA

# Datos de entrenamiento.
data = {
    'text': [
        "Claro, me encantó esperar una hora para que al final cancelaran mi pedido. La mejor experiencia.",
        "¿De verdad creen que este servicio justifica el precio que cobran?",
        "Qué buena idea hacer mantenimiento a la web en hora punta. Simplemente genial.",
        "Recibir un producto roto es mi definición de un servicio de cinco estrellas. Gracias.",
        "El producto es fantástico, superó todas mis expectativas.",
        "No está mal, pero podría mejorar en algunos aspectos."
    ],
    'label': [
        1, # Negative
        1, # Negative
        1, # Negative
        1, # Negative
        4, # Very Positive
        2  # Neutral
    ]
}

df = pd.DataFrame(data)

# Guarda el dataset en un archivo CSV, forzando las comillas en todos los campos
df.to_csv(
    "mi_dataset.csv", 
    index=False, 
    quoting=csv.QUOTE_ALL # <--- AÑADE ESTA LÍNEA
)

print("Dataset creado y guardado como 'mi_dataset.csv' con formato robusto.")
