import pandas as pd
import sqlite3
import os
from transformers import pipeline
import torch # Transformers necesita que torch o tensorflow esté instalado

# --- Configuración ---
# Archivos de entrada y salida
db_original_path = 'analisis_sentimientos.db'
db_nueva_path = 'analisis_sentimientos_con_resultados.db'

# Nombres de la tabla y la columna a analizar
table_name = 'comentarios'
columna_a_analizar = 'title' # ¡Importante! Cambia esto si la columna tiene otro nombre

# Modelo de Hugging Face para el análisis de 5 sentimientos
modelo_transformers = "tabularisai/multilingual-sentiment-analysis"


def analizar_base_de_datos():
    """
    Lee datos de una base de datos, analiza una columna de texto
    y guarda los resultados en una nueva base de datos.
    """
    # 1. Verificar que la base de datos original exista
    if not os.path.exists(db_original_path):
        print(f"Error: No se encontró la base de datos '{db_original_path}'.")
        return

    print("Paso 1 de 5: Cargando el modelo de análisis de sentimientos...")
    # Cargar el pipeline de Transformers. Se recomienda usar GPU si está disponible.
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("text-classification", model=modelo_transformers, device=device)
    print("Modelo cargado exitosamente.")

    try:
        # 2. Conectar a la DB original y cargar los datos en un DataFrame de pandas
        print(f"\nPaso 2 de 5: Leyendo la tabla '{table_name}' de '{db_original_path}'...")
        conn_original = sqlite3.connect(db_original_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn_original)
        conn_original.close()
        print(f"Se cargaron {len(df)} filas.")

        # 3. Preparar los textos para el análisis
        print(f"\nPaso 3 de 5: Preparando textos de la columna '{columna_a_analizar}'...")
        # Asegurarse de que no haya valores nulos (NaN) que puedan dar error
        textos_para_analizar = df[columna_a_analizar].fillna('').tolist()

        # 4. Ejecutar el análisis de sentimientos
        # Esto puede tardar varios minutos dependiendo de la cantidad de datos y del hardware (CPU/GPU)
        print(f"\nPaso 4 de 5: Analizando {len(textos_para_analizar)} comentarios... (esto puede tardar)")
        resultados = sentiment_pipeline(textos_para_analizar, batch_size=8) # batch_size para optimizar
        print("Análisis completado.")

        # 5. Añadir los resultados al DataFrame
        print("\nPaso 5 de 5: Agregando nuevas columnas y guardando la nueva base de datos...")
        df['sentimiento_predicho'] = [res['label'] for res in resultados]
        df['confianza_prediccion'] = [res['score'] for res in resultados]

        # Guardar el DataFrame modificado en una nueva base de datos
        conn_nueva = sqlite3.connect(db_nueva_path)
        df.to_sql(table_name, conn_nueva, if_exists='replace', index=False)
        conn_nueva.close()

        print("\n--- ¡Proceso completado! ---")
        print(f"Se ha creado una nueva base de datos '{db_nueva_path}'")
        print("Contiene la tabla original más las columnas 'sentimiento_predicho' y 'confianza_prediccion'.")

    except KeyError:
        print(f"\nError: La columna '{columna_a_analizar}' no se encontró en la tabla '{table_name}'.")
        print("Por favor, verifica el nombre de la columna en la sección de configuración del script.")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")

if __name__ == "__main__":
    analizar_base_de_datos()
