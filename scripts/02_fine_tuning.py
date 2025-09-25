from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

# --- 1. Cargar y Tokenizar ---

# Cargar tu dataset desde el archivo CSV
dataset = load_dataset('csv', data_files='mi_dataset.csv', split='train')

# Nombre del modelo base que vamos a mejorar
model_name = "tabularisai/multilingual-sentiment-analysis"

# Cargar el tokenizador para pre-procesar el texto
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Función para tokenizar el dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Aplicar la tokenización a todo el dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)


# --- 2. Cargar el Modelo Base ---

# Definimos el mapeo de ID a etiqueta para que el modelo lo entienda
id2label = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
label2id = {"Very Negative": 0, "Negative": 1, "Neutral": 2, "Positive": 3, "Very Positive": 4}

# Cargar el modelo pre-entrenado, especificando el número de etiquetas y los mapeos
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)


# --- 3. Configurar el Entrenamiento ---

# Definir los argumentos del entrenamiento
training_args = TrainingArguments(
    output_dir="./resultados_finetuning", # Carpeta donde se guardará el modelo
    num_train_epochs=4,                  # Número de veces que se entrenará con todo el dataset
    per_device_train_batch_size=4,       # Número de ejemplos por lote
    learning_rate=2e-5,                  # Tasa de aprendizaje
    weight_decay=0.01,
    logging_steps=1,
)

# Crear el objeto Trainer, que maneja todo el proceso de entrenamiento
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # No tenemos un set de evaluación, pero en un proyecto real es crucial
    # eval_dataset=eval_dataset 
)

# --- 4. ¡Iniciar el Fine-Tuning! ---
print("Iniciando el proceso de fine-tuning...")
trainer.train()
print("¡Fine-tuning completado!")

# Guardar el modelo afinado y el tokenizador en la carpeta de resultados
trainer.save_model("./modelo_afinad")
tokenizer.save_pretrained("./modelo_afinad")
print("Modelo afinado guardado en la carpeta './modelo_afinad'")
