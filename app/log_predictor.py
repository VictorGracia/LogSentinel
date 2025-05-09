import os
import numpy as np
from keras.api.models import load_model
from keras.api.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
MODEL_PATH = os.path.join('models', 'log_model.keras')

# Cargar modelo
model = load_model(MODEL_PATH)

# Función para predecir el nivel del log
def predict_log_level(message):
    message = message.lower().strip()
    vectorizer = TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=SEQUENCE_LENGTH)
    X_input = vectorizer([message])
    prediction = model.predict(X_input)
    label_encoder = LabelEncoder()
    predicted_class = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class])[0]

# Predicción en lote
def predict_logs(logs):
    results = []
    for log in logs:
        prediction = predict_log_level(log)
        results.append({"log": log, "predicted_level": prediction})
    return {"predictions": results}
