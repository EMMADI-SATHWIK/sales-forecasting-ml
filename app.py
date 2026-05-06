from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Sales Forecast API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)