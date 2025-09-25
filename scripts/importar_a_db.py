import pandas as pd
import sqlite3
import os

# --- Configuración ---
# Nombre del archivo CSV que quieres importar
csv_file_path = 'copiaPremios.csv' 
# Nombre del archivo de la base de datos que se creará
db_file_path = 'analisis_sentimientos.db' 
# Nombre que tendrá la tabla dentro de la base de datos
table_name = 'comentarios'

def importar_csv_a_db():
    """
    Lee un archivo CSV y lo importa a una tabla en una base de datos SQLite.
    Si la tabla ya existe, la reemplaza con los nuevos datos.
    """
    # 1. Verificar que el archivo CSV exista
    if not os.path.exists(csv_file_path):
        print(f"Error: El archivo '{csv_file_path}' no se encontró.")
        print("Por favor, asegúrate de que el archivo CSV esté en la misma carpeta que este script.")
        return

    try:
        # 2. Leer el archivo CSV usando pandas
        # Pandas se encarga de interpretar las comas y separar todo en columnas.
        print(f"Leyendo el archivo '{csv_file_path}'...")
        # Se intenta leer con codificación 'latin-1' que es común en datos en español
        df = pd.read_csv(csv_file_path, encoding='latin-1')
        print("Archivo CSV leído exitosamente.")

        # Opcional: Limpiar nombres de columnas (reemplazar espacios por guiones bajos)
        df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        # 3. Conectar a la base de datos SQLite
        # Si el archivo .db no existe, se creará automáticamente.
        conn = sqlite3.connect(db_file_path)
        print(f"Conectado a la base de datos '{db_file_path}'.")

        # 4. Usar la función to_sql de pandas para pasar los datos a la DB
        # if_exists='replace' borrará la tabla si ya existe y la creará de nuevo.
        # Esto es útil para volver a cargar los datos sin duplicados.
        print(f"Importando datos a la tabla '{table_name}'...")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print("¡Datos importados exitosamente!")

        # 5. Verificar los datos (opcional)
        print("\n--- Verificación ---")
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"La tabla '{table_name}' ahora tiene {row_count} filas.")
        print(f"La columna con los comentarios a analizar probablemente es 'Content'.")


    except Exception as e:
        print(f"Ocurrió un error durante el proceso: {e}")
    finally:
        # 6. Cerrar la conexión a la base de datos
        if 'conn' in locals() and conn:
            conn.close()
            print("Conexión a la base de datos cerrada.")

if __name__ == "__main__":
    importar_csv_a_db()
