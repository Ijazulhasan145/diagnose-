import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pickled model
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    
    
    try:
        df1 = pd.read_csv('Symptom-severity.csv')
       
        val1 = request.form.get('dropdown1').strip().lower()
        val2 = request.form.get('dropdown2').strip().lower()
        val3 = request.form.get('dropdown3').strip().lower()

        print("Hello g")


        symptom_weight_map = dict(zip(df1['Symptom'].str.strip().str.lower(), df1['weight']))

        weights = [
            symptom_weight_map.get(val1, val1),
            symptom_weight_map.get(val2, val2),
            symptom_weight_map.get(val3, val3)
        ]

        input_data = weights + [0] * (17 - len(weights))
        input_array = np.array(input_data).reshape(1, -1)  

        prediction = model.predict(input_array)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'error': 'Invalid input values. Please select valid numeric options.'}), 400

    

if __name__ == '__main__':
    app.run(debug=True)
