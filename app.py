from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# --- MODEL LOADING ---
# Ensure your trained model is named 'model.pkl' in the same folder.
# If you haven't saved your model yet, uncomment these lines later:
# model = pickle.load(open('model.pkl', 'rb'))

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
            # 1. Extract data from the predict.html form fields
            step = int(request.form['step'])
            type_val = request.form['type']
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])

            # 2. Categorical Encoding for 'Type'
            # (Matches common datasets like PaySim: CASH_OUT=1, PAYMENT=2, etc.)
            type_map = {
                'CASH_OUT': 1, 
                'PAYMENT': 2, 
                'CASH_IN': 3, 
                'TRANSFER': 4, 
                'DEBIT': 5
            }
            type_encoded = type_map.get(type_val, 0)

            # 3. Prepare the feature array for the model
            features = np.array([[
                step, 
                type_encoded, 
                amount, 
                oldbalanceOrg, 
                newbalanceOrig, 
                oldbalanceDest, 
                newbalanceDest
            ]])

            # 4. Make Prediction
            # 1 = Fraud, 0 = Safe
            # prediction = model.predict(features)[0]
            
            # Placeholder: Set this to 1 or 0 to test your submit.html UI
            prediction = 1 

            # 5. Send result to submit.html template
            return render_template('submit.html', prediction=prediction)

        except Exception as e:
            return f"Error in processing: {str(e)}"

if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(debug=True)