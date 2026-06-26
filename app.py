import os
from flask import Flask, render_template, request, jsonify
from model.predict import predict_diabetes, get_model_info

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    # POST request: predict diabetes
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Run prediction
        result = predict_diabetes(data)
        return jsonify(result), 200
        
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except FileNotFoundError as fnf_err:
        return jsonify({"error": str(fnf_err)}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/about')
def about():
    # Attempt to load model info to show on the About page
    try:
        model_info = get_model_info()
    except Exception:
        model_info = {"model_name": "Model not trained yet", "metrics": {}}
    return render_template('about.html', model_info=model_info)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
