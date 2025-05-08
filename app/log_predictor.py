import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
MODEL_PATH = os.path.join('models', 'log_model.keras')
CSV_PATH = os.path.join('data', 'application_logs.csv')

# Cargar modelo
model = load_model(MODEL_PATH)

# Crear y adaptar vectorizador y encoder
df = pd.read_csv(CSV_PATH)
df = df[['message_details', 'log_level']].dropna()
df['message_details'] = df['message_details'].astype(str).str.lower().str.strip()

vectorizer = TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=SEQUENCE_LENGTH)
vectorizer.adapt(df['message_details'])

label_encoder = LabelEncoder()
label_encoder.fit(df['log_level'])

# Predicción individual
def predict_log_level(message):
    message = message.lower().strip()
    X_input = vectorizer([message])
    prediction = model.predict(X_input)
    predicted_class = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class])[0]

# Predicción en lote (para API)
def predict_logs(logs):
    results = []
    for log in logs:
        prediction = predict_log_level(log)
        results.append({"log": log, "predicted_level": prediction})
    return {"predictions": results}
