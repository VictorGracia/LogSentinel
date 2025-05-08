import pandas as pd
import numpy as np
import os
import datetime
from flask import request, jsonify
from keras.api.models import load_model
from keras.api.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
MODEL_PATH = os.path.join('models', 'log_model.keras')

model = load_model(MODEL_PATH)

def analyze_uploaded_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    df = pd.read_csv(file)
    df = df[['message_details', 'log_level']].dropna()
    df['message_details'] = df['message_details'].astype(str).str.lower().str.strip()

    # Simular timestamps si no existen
    start_time = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    df['timestamp'] = [start_time + datetime.timedelta(minutes=i*5) for i in range(len(df))]

    # Vectorización y predicción
    vectorizer = TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=SEQUENCE_LENGTH)
    vectorizer.adapt(df['message_details'])

    label_encoder = LabelEncoder()
    label_encoder.fit(df['log_level'])

    sequences = vectorizer(df['message_details']).numpy()
    predictions = model.predict(sequences)
    predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    df['predicted_class'] = predicted_classes
    df['confidence'] = predictions.max(axis=1)

    # Anomalías
    anomalies = df[df['confidence'] < 0.75]
    anomaly_records = []
    for idx, row in anomalies.iterrows():
        message = row['message_details']
        anomaly_type = "unauthorized_access_attempt" if "unauthorized" in message or "failed login" in message else "unknown_log_behavior"
        anomaly_records.append({
            "line": int(idx),
            "timestamp": row['timestamp'].isoformat() + 'Z',
            "type": anomaly_type,
            "severity": "medium" if row['confidence'] > 0.5 else "high",
            "score": round(row['confidence'], 3),
            "message": message,
            "rawEntry": row.to_dict()
        })

    last_log_time = df['timestamp'].max()
    expected_end_time = start_time + datetime.timedelta(hours=24)
    premature = last_log_time < expected_end_time - datetime.timedelta(minutes=30)

    summary = {
        "totalEntries": len(df),
        "anomaliesDetected": len(anomalies),
        "anomalyRate": round((len(anomalies) / len(df)) * 100, 2),
        "severityDistribution": {
            "low": int((anomalies['confidence'] >= 0.6).sum()),
            "medium": int(((anomalies['confidence'] < 0.6) & (anomalies['confidence'] >= 0.4)).sum()),
            "high": int((anomalies['confidence'] < 0.4).sum())
        },
        "mostFrequentAnomalyType": most_frequent_anomaly_type(anomaly_records),
        "prematureTerminationDetected": premature,
        "terminationDetails": {
            "expectedDuration": "24h",
            "actualDuration": str(last_log_time - start_time),
            "lastLogTimestamp": last_log_time.isoformat() + 'Z',
            "expectedEndOfDay": expected_end_time.isoformat() + 'Z',
            "message": "Log file ended prematurely, possible crash or logging failure." if premature else "Log file ended as expected."
        } if premature else {}
    }

    return jsonify({
        "fileName": file.filename,
        "analysisTimestamp": datetime.datetime.utcnow().isoformat() + 'Z',
        "summary": summary,
        "anomalies": anomaly_records,
        "model": {
            "version": "1.0.3",
            "confidenceThreshold": 0.75
        }
    })

def most_frequent_anomaly_type(anomalies):
    from collections import Counter
    types = [a['type'] for a in anomalies]
    return Counter(types).most_common(1)[0][0] if types else "unknown"
