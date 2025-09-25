from transformers import pipeline

# Comentario que antes fue clasificado incorrectamente como "Positive"
sarcastic_comment = "Claro, me encantó esperar una hora para que al final cancelaran mi pedido. La mejor experiencia."

# Cargar un pipeline usando NUESTRO modelo afinado guardado en la carpeta local
sentiment_analyzer_afinad = pipeline(
    "text-classification",
    model="./modelo_afinad" 
)

# Analizar el comentario con el nuevo modelo
result = sentiment_analyzer_afinad(sarcastic_comment)

print(f"Comentario: '{sarcastic_comment}'")
print(f"Resultado del modelo afinado: {result}")
