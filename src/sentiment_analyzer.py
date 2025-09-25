# 📁 src/sentiment_analyzer.py

import pandas as pd
from transformers import pipeline

def run_sentiment_analysis():
    """
    Carga un modelo de sentiment analysis y lo prueba con ejemplos hardcodeados.
    """
    print("Iniciando el script de análisis de sentimientos...")

    # 1. Cargar el pipeline con el modelo específico
    try:
        print("Cargando el modelo de Transformers...")
        sentiment_analyzer = pipeline(
            "text-classification",
            model="tabularisai/multilingual-sentiment-analysis"
        )
        print("¡Modelo cargado exitosamente!")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # 2. Definir los comentarios de ejemplo (hardcodeados)
    comentarios = [
        "El producto es fantástico, superó todas mis expectativas.",
        "El servicio fue terrible y la espera interminable.",
        "No está mal, pero podría mejorar en algunos aspectos.",
        "This is a neutral comment.",
    ]
    print(f"\nAnalizando {len(comentarios)} comentarios de ejemplo...")

    # 3. Realizar el análisis de sentimientos
    try:
        resultados = sentiment_analyzer(comentarios)
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        return

    # 4. Formatear y mostrar los resultados
    df = pd.DataFrame({
        'Comentario': comentarios,
        'Sentimiento': [res['label'] for res in resultados],
        'Confianza': [round(res['score'], 4) for res in resultados]
    })

    print("\n--- Resultados del Análisis ---")
    print(df.to_string(index=False))
    print("\n------------------------------")


if __name__ == "__main__":
    run_sentiment_analysis()
    print("\n✅ Script ejecutado exitosamente. El workflow funciona.")
