from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import xgboost as xgb
import pandas as pd

# Setup Flask and point to correct templates folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

# Load the trained model and features
model = xgb.XGBClassifier()
model.load_model(os.path.join('../model', 'xgb_model.json'))
features = joblib.load(os.path.join('../model', 'model_features.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Collect form inputs
        age = int(request.form['Age'])
        gender = request.form['Gender']
        marital_status = request.form['MaritalStatus']
        overtime = request.form['OverTime']
        monthly_income = float(request.form['MonthlyIncome'])
        job_satisfaction = int(request.form['JobSatisfaction'])
        years_at_company = int(request.form['YearsAtCompany'])
        jobrole = request.form['JobRole']
        business_travel = request.form['BusinessTravel']

        # 2. Start building input dictionary
        data_dict = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'JobSatisfaction': job_satisfaction,
            'YearsAtCompany': years_at_company,

            'Gender_Male': 1 if gender == 'Male' else 0,
            'Gender_Female': 1 if gender == 'Female' else 0,

            'MaritalStatus_Married': 1 if marital_status == 'Married' else 0,
            'MaritalStatus_Single': 1 if marital_status == 'Single' else 0,
            'MaritalStatus_Divorced': 1 if marital_status == 'Divorced' else 0,

            'OverTime_Yes': 1 if overtime == 'Yes' else 0,
            'OverTime_No': 1 if overtime == 'No' else 0,
        }

        # 3. One-hot encode Job Role
        for role in [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Human Resources", "Research Director"
        ]:
            data_dict[f"JobRole_{role}"] = 1 if jobrole == role else 0

        # 4. One-hot encode Business Travel
        for travel in ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]:
            data_dict[f"BusinessTravel_{travel}"] = 1 if business_travel == travel else 0

        # 5. Fill missing features with 0 to match model input
        for feat in features:
            if feat not in data_dict:
                data_dict[feat] = 0

        # 6. Create DataFrame in exact order of features
        X_input = pd.DataFrame([data_dict])[features]

        # 7. Predict
        print("INPUT TO MODEL:")
        print(X_input)

        prediction = model.predict(X_input)[0]
        result = "✅ Likely to Leave (Attrition)" if prediction == 1 else "🟢 Not Likely to Leave"

        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
