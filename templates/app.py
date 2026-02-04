from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# --- 1. MODEL LOADING ---
# Ensure your train_model.py has finished creating 'payments.pkl'
model = pickle.load(open('payments.pkl', 'rb'))

@app.route('/')
def home():
    """Renders the Home/Objective page."""
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    """Renders the input form page."""
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    """Captures form data, runs the model, and shows submit.html."""
    if request.method == 'POST':
        try:
            # 2. Extract data from the predict.html form fields
            step = int(request.form['step'])
            type_val = request.form['type']
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])

            # 3. Categorical Encoding for 'Type'
            type_map = {
                'CASH_OUT': 1, 
                'PAYMENT': 2, 
                'CASH_IN': 3, 
                'TRANSFER': 4, 
                'DEBIT': 5
            }
            type_encoded = type_map.get(type_val, 0)

            # 4. Prepare the feature array for the model
            features = np.array([[
                step, 
                type_encoded, 
                amount, 
                oldbalanceOrg, 
                newbalanceOrig, 
                oldbalanceDest, 
                newbalanceDest
            ]])

            # 5. Make REAL Prediction (Removed the static placeholder 'prediction = 1')
            prediction = model.predict(features)[0]

            # 6. Send actual result to submit.html template
            return render_template('submit.html', prediction=int(prediction))

        except Exception as e:
            return f"Error in processing: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
