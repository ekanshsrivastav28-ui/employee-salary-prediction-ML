from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/salary_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    exp = np.array([[data['YearsExperience']]])
    prediction = model.predict(exp)
    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
