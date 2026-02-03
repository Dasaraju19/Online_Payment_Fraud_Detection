from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read values from form
    features = [float(x) for x in request.form.values()]

    # Pad features to match model input size (11 features)
    REQUIRED_FEATURES = 11
    if len(features) < REQUIRED_FEATURES:
        features.extend([0] * (REQUIRED_FEATURES - len(features)))

    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
