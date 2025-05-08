from flask import Flask, request, jsonify
from app.model_trainer import train_log_model
from app.log_predictor import predict_logs
from app.csv_analyzer import analyze_uploaded_csv

def create_app():
    app = Flask(__name__)

    @app.route('/train_model', methods=['POST'])
    def train_model():
        try:
            return train_log_model()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            logs = data.get("logs", [])
            return jsonify(predict_logs(logs))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/analyze_csv', methods=['POST'])
    def analyze():
        return analyze_uploaded_csv()

    return app
