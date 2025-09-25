# Crea este script para generar tu archivo CSV, o créalo manualmente.
import pandas as pd

# Datos de entrenamiento. Incluimos el ejemplo sarcástico y lo etiquetamos correctamente.
data = {
    'text': [
        # El ejemplo de sarcasmo que falló, ahora etiquetado como Negativo (1)
        "Claro, me encantó esperar una hora para que al final cancelaran mi pedido. La mejor experiencia.",
        # La pregunta retórica, ahora etiquetada como Negativo (1)
        "¿De verdad creen que este servicio justifica el precio que cobran?",
        # Agregamos más ejemplos de sarcasmo para reforzar el aprendizaje
        "Qué buena idea hacer mantenimiento a la web en hora punta. Simplemente genial.",
        "Recibir un producto roto es mi definición de un servicio de cinco estrellas. Gracias.",
        # Agregamos ejemplos 'normales' para que el modelo no olvide lo que ya sabe
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

# Guarda el dataset en un archivo CSV
df.to_csv("mi_dataset.csv", index=False)

print("Dataset creado y guardado como 'mi_dataset.csv'")
