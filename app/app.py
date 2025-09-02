#Flasl API for Model Inference
from flask import Flask, request, jsonify
import pickle
import numpy as np

#Load the model
with open('app/diabetes_model.pkl', 'rb') as file:
    scaler, model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return "Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    output = int(prediction[0])
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000,debug=True)