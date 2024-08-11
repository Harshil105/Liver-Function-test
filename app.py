from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load your machine learning model
with open('D:/AIML/liver function test/random_forest_model.sav', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()

    # Extract the array from the JSON
    input_array = data['data']
    # Make predictions
    prediction = model.predict(input_array)[0]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
