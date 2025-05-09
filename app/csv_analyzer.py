import pandas as pd
import numpy as np
import os
import datetime
import re
from flask import request, jsonify
from keras.api.models import load_model
from keras.api.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
MODEL_PATH = os.path.join('models', 'log_model.keras')

model = load_model(MODEL_PATH)

# Expresiones regulares de anomalías conocidas
ANOMALY_PATTERNS = [
    (r'unauthorized access|failed login', 'unauthorized_access_attempt', 'high'),
    (r'system crash|segmentation fault|kernel panic', 'system_crash', 'critical'),
    (r'disk space.*(full|low|below)', 'disk_space_issue', 'medium'),
    (r'memory usage.*(exceeded|leak)', 'memory_leak', 'high'),
    (r'network timeout|failed to fetch|connection timeout', 'network_timeout', 'medium'),
    (r'database connection.*(failed|timeout)', 'database_connection_error', 'high'),
]

ALLOWLIST_PATTERNS = [
    r'user logged in successfully',
    r'file uploaded successfully',
    r'connection established',
    r'database connection established successfully',
]

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
    anomalies = []
    for idx, row in df.iterrows():
        message = row['message_details']

        # Saltar mensajes explícitamente normales
        if any(re.search(p, message) for p in ALLOWLIST_PATTERNS):
            continue

        matched = False
        for pattern, anomaly_type, severity in ANOMALY_PATTERNS:
            if re.search(pattern, message):
                matched = True
                anomalies.append({
                    "line": int(idx),
                    "timestamp": row['timestamp'].isoformat() + 'Z',
                    "type": anomaly_type,
                    "severity": severity,
                    "score": round(row['confidence'], 3),
                    "message": message,
                    "rawEntry": row.to_dict()
                })
                break

        if not matched:
            continue  # ya no marcamos como unknown_log_behavior por defecto

    last_log_time = df['timestamp'].max()
    expected_end_time = start_time + datetime.timedelta(hours=24)
    premature = last_log_time < expected_end_time - datetime.timedelta(minutes=30)

    # Calcular distribución de severidad
    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for a in anomalies:
        sev = a['severity']
        if sev in severity_counts:
            severity_counts[sev] += 1

    summary = {
        "totalEntries": len(df),
        "anomaliesDetected": len(anomalies),
        "anomalyRate": round((len(anomalies) / len(df)) * 100, 2),
        "severityDistribution": severity_counts,
        "mostFrequentAnomalyType": most_frequent_anomaly_type(anomalies),
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
        "anomalies": anomalies,
        "model": {
            "version": "1.0.3",
            "confidenceThreshold": 0.75
        }
    })

def most_frequent_anomaly_type(anomalies):
    from collections import Counter
    types = [a['type'] for a in anomalies]
    return Counter(types).most_common(1)[0][0] if types else "unknown"
