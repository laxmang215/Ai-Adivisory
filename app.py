from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        # Load the trained model
        model = LinearRegression()
        model.load('trained_model.joblib')  # Replace with your actual model file

        # Make a prediction
        prediction = model.predict([[feature1, feature2, feature3]])

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)