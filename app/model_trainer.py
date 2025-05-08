import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, Dropout, TextVectorization
from keras.api.utils import to_categorical

# ========= Configuración =========
CSV_PATH = os.path.join('data', 'application_logs.csv')  # Asegúrate de que este archivo exista
MODEL_PATH = os.path.join('models', 'log_model.keras')
VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
# =================================

def train_log_model():
    print("📥 Cargando archivo:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    if 'message_details' not in df.columns or 'log_level' not in df.columns:
        raise ValueError("❌ El CSV debe tener las columnas 'message_details' y 'log_level'.")

    df = df[['message_details', 'log_level']].dropna()
    df['message_details'] = df['message_details'].astype(str).str.lower().str.strip()

    print("🧹 Limpieza y preprocesamiento de texto...")
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH
    )
    vectorizer.adapt(df['message_details'])
    X = vectorizer(df['message_details']).numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['log_level'])
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Construyendo modelo LSTM...")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=SEQUENCE_LENGTH),
        LSTM(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("🚀 Entrenando modelo...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    print(f"✅ Modelo entrenado y guardado en: {MODEL_PATH}")
    return model, vectorizer, label_encoder


# ========= Soporte para ejecución directa =========
if __name__ == '__main__':
    try:
        train_log_model()
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
