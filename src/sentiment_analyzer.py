# 📁 src/sentiment_analyzer.py

import pandas as pd
from transformers import pipeline

def run_sentiment_analysis():
    """
    Carga un modelo de sentiment analysis y lo prueba con ejemplos complejos.
    """
    print("Iniciando el script de análisis de sentimientos con ejemplos difíciles...")

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

    # 2. Definir los comentarios de ejemplo complejos (hardcodeados)
    #    Estos ejemplos incluyen sarcasmo, sentimientos mixtos y lenguaje sutil.
    comentarios = [
        # Sarcasmo: Usa palabras positivas para expresar una idea negativa.
        "Claro, me encantó esperar una hora para que al final cancelaran mi pedido. La mejor experiencia.",
        
        # Sentimiento Mixto: Contiene elementos positivos y negativos.
        "Aunque la trama de la película es original y atrapante, la actuación del protagonista es pésima.",
        
        # Sutileza / Neutralidad compleja: No expresa un sentimiento claro y directo.
        "El libro no es malo, pero definitivamente no es la obra maestra que la crítica aclamaba.",

        # Condicional negativo: Expresa una condición que no se cumplió y resultó en una mala experiencia.
        "El hotel habría sido perfecto si la habitación no hubiera estado sucia y el aire acondicionado roto.",
        
        # Pregunta retórica con carga negativa.
        "¿De verdad creen que este servicio justifica el precio que cobran?"
    ]
    print(f"\nAnalizando {len(comentarios)} comentarios complejos...")

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

    print("\n--- Resultados del Análisis (Ejemplos Difíciles) ---")
    print(df.to_string(index=False))
    print("\n----------------------------------------------------")


if __name__ == "__main__":
    run_sentiment_analysis()
    print("\n✅ Script ejecutado exitosamente. El workflow funciona.")
